"""Run the frozen DABench explore/test split with DSLighting ReAct."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .protocol import (
    PAPER_DISCOVERY_ROUNDS,
    PAPER_EVALUATION_TEMPERATURE,
    PAPER_EXPLORE_TEMPERATURE,
    PAPER_REACT_MAX_STEPS,
    PAPER_TRAJECTORIES_PER_TASK,
    validate_round_index,
    validate_sample_index,
)
from .split import DEFAULT_FAMILY_MANIFEST, DABenchSplit, load_dabench_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "releases" / "dabench_perception_clean_v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments" / "datacope_dabench" / "runs"
DEFAULT_MODEL = "openai/DeepSeek-V4-Flash"


@dataclass(frozen=True)
class BenchmarkRunSpec:
    phase: str
    round_index: int | None
    sample_index: int
    task_ids: tuple[str, ...]
    run_name: str
    data_root: Path
    workspace_base: Path
    workspace_root: Path
    log_dir: Path
    model: str
    temperature: float
    skill_file: Path | None
    skill_sha256: str | None
    split: DABenchSplit


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_run_spec(
    *,
    phase: str,
    sample_index: int = 0,
    round_index: int | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    family_manifest: Path = DEFAULT_FAMILY_MANIFEST,
    run_root: Path = DEFAULT_RUN_ROOT,
    run_name: str | None = None,
    model: str = DEFAULT_MODEL,
    skill_file: Path | None = None,
) -> BenchmarkRunSpec:
    """Resolve all experiment inputs without starting a benchmark."""

    split = load_dabench_split(family_manifest)
    task_ids = split.task_ids(phase)

    resolved_round: int | None
    if phase == "explore":
        resolved_round = validate_round_index(0 if round_index is None else round_index)
        validate_sample_index(sample_index)
        temperature = PAPER_EXPLORE_TEMPERATURE
    else:
        if round_index is not None:
            raise ValueError("round_index applies only to the explore phase")
        if sample_index != 0:
            raise ValueError("Held-out evaluation is a single deterministic run")
        resolved_round = None
        temperature = PAPER_EVALUATION_TEMPERATURE

    resolved_skill: Path | None = None
    skill_sha256: str | None = None
    if skill_file is not None:
        if phase == "explore" and resolved_round == 0:
            raise ValueError("Explore round 0 must run without a skill")
        resolved_skill = Path(skill_file).expanduser().resolve()
        if not resolved_skill.is_file():
            raise ValueError(f"Skill file does not exist: {resolved_skill}")
        try:
            skill_text = resolved_skill.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read skill file {resolved_skill}: {exc}") from exc
        if not skill_text:
            raise ValueError(f"Skill file is empty: {resolved_skill}")
        skill_sha256 = _file_sha256(resolved_skill)
    if phase == "explore" and resolved_round and resolved_skill is None:
        raise ValueError(f"Explore round {resolved_round} requires the previous round skill")

    resolved_data = Path(data_root).expanduser().resolve()
    if not resolved_data.is_dir():
        raise ValueError(f"DABench data root does not exist: {resolved_data}")
    missing = [task_id for task_id in task_ids if not (resolved_data / task_id).is_dir()]
    if missing:
        raise ValueError(f"DABench data root is missing split tasks: {missing[:5]}")

    arm = "skill" if resolved_skill else "baseline"
    default_name = (
        f"dabench_datacope_explore_round_{resolved_round}_sample_{sample_index:02d}"
        if phase == "explore"
        else f"dabench_datacope_test_{arm}"
    )
    resolved_name = str(run_name or default_name).strip()
    if not resolved_name or "/" in resolved_name or "\\" in resolved_name:
        raise ValueError("run_name must be one non-empty path component")

    resolved_run_root = Path(run_root).expanduser().resolve()
    workspace_base = resolved_run_root / "workspaces"
    resolved_model = str(model).strip()
    if not resolved_model:
        raise ValueError("model must be non-empty")
    return BenchmarkRunSpec(
        phase=phase,
        round_index=resolved_round,
        sample_index=sample_index,
        task_ids=task_ids,
        run_name=resolved_name,
        data_root=resolved_data,
        workspace_base=workspace_base,
        workspace_root=workspace_base / resolved_name,
        log_dir=resolved_run_root / "benchmarks" / resolved_name,
        model=resolved_model,
        temperature=temperature,
        skill_file=resolved_skill,
        skill_sha256=skill_sha256,
        split=split,
    )


def require_bubblewrap() -> None:
    """Fail closed when the configured sandbox cannot enforce isolation."""

    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("bubblewrap is required for this experiment")
    completed = subprocess.run(
        [
            executable,
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--unshare-net",
            "--",
            "/bin/true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"bubblewrap preflight failed: {detail}")


def run_dabench_react(
    spec: BenchmarkRunSpec,
    *,
    max_steps: int = PAPER_REACT_MAX_STEPS,
    max_concurrency: int = 50,
    llm_max_concurrency: int = 50,
    timeout_seconds: int = 2 * 60 * 60,
) -> int:
    """Run one explore sample or one held-out baseline/skill condition."""

    if min(max_steps, max_concurrency, llm_max_concurrency, timeout_seconds) <= 0:
        raise ValueError("runtime limits must be positive")
    for path in (spec.workspace_root, spec.log_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Experiment output must be empty or new: {path}")

    require_bubblewrap()
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    from dslighting import configure_logging
    from dslighting.api.benchmark import DSBenchmark
    from dslighting.benchmark.core.source_catalog import get_benchmark_source_catalog
    from dslighting.core.config.builder import ConfigBuilder

    agent_runtime: dict[str, object] = {"max_steps": max_steps}
    if spec.skill_file is not None:
        agent_runtime["skill_path"] = str(spec.skill_file)

    config = ConfigBuilder().build_config(
        workflow="react",
        model=spec.model,
        temperature=spec.temperature,
        workspace_dir=str(spec.workspace_base),
        run_name=spec.run_name,
        keep_workspace=True,
        keep_workspace_on_failure=True,
        sandbox={
            "backend": "local",
            "local_isolation": "bubblewrap",
            "environment_policy": "allowlist",
            "network_policy": "disabled",
            "timeout": timeout_seconds,
        },
        data_analysis={"cache_enabled": False},
        agent_runtime=agent_runtime,
    )
    config.run.parameters.update(
        {
            "datacope_split_id": spec.split.split_id,
            "datacope_source_manifest_sha256": spec.split.source_manifest_sha256,
            "datacope_phase": spec.phase,
            "datacope_round_index": spec.round_index,
            "datacope_sample_index": spec.sample_index,
            "datacope_discovery_rounds": PAPER_DISCOVERY_ROUNDS,
            "datacope_trajectories_per_task": PAPER_TRAJECTORIES_PER_TASK,
            "datacope_temperature": spec.temperature,
            "datacope_max_steps": max_steps,
            "datacope_skill_path": str(spec.skill_file) if spec.skill_file else None,
            "datacope_skill_sha256": spec.skill_sha256,
        }
    )
    config.run.dag_runtime.node_timeout_seconds = float(timeout_seconds)
    config.scheduler.scheduler_policy = "balanced"
    config.scheduler.max_concurrency = max_concurrency
    config.scheduler.llm_max_concurrency = llm_max_concurrency
    config.llm.max_concurrent_per_key = llm_max_concurrency

    configure_logging(
        level="INFO",
        trace_llm=False,
        output_dir=str(spec.log_dir / "debug_logs"),
        force=True,
    )

    source = get_benchmark_source_catalog().get_source("dabench")
    benchmark = DSBenchmark(
        benchmark_type="dabench",
        exp_name=spec.run_name,
        data_dir=str(spec.data_root),
        vendor_comp_dir=str(source.registry_root),
        competitions=list(spec.task_ids),
    )

    print(f"phase={spec.phase}")
    print(f"round_index={spec.round_index if spec.round_index is not None else 'none'}")
    print(f"sample_index={spec.sample_index}")
    print(f"tasks={len(spec.task_ids)}")
    print(f"model={spec.model}")
    print(f"temperature={spec.temperature}")
    print(f"skill_sha256={spec.skill_sha256 or 'none'}")
    print(f"workspace_root={spec.workspace_root}")
    print(f"log_dir={spec.log_dir}")
    result = benchmark.run(config=config, log_path=str(spec.log_dir), verbose=True)
    print(f"results_path={getattr(result, 'results_path', '')}")
    print(f"metadata_path={getattr(result, 'metadata_path', '')}")
    return 0


__all__ = [
    "BenchmarkRunSpec",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MODEL",
    "DEFAULT_RUN_ROOT",
    "require_bubblewrap",
    "resolve_run_spec",
    "run_dabench_react",
]
