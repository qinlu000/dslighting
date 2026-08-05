"""Run the frozen MoSciBench split with DSLighting's built-in ReAct workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.datacope_dabench.runner import require_bubblewrap

from .protocol import (
    EXPLORE_TEMPERATURE,
    TEST_TEMPERATURE,
    validate_round_index,
    validate_sample_index,
)
from .split import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_MANIFEST, MoSciSplit, load_split, task_family

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path(__file__).with_name("runs")
DEFAULT_MODEL = "openai/DeepSeek-V4-Flash"
DEFAULT_MAX_STEPS = 10
DEFAULT_TASK_TIMEOUT_SECONDS = 2 * 60 * 60


@dataclass(frozen=True)
class RunSpec:
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
    split: MoSciSplit


def resolve_run_spec(
    *,
    phase: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
    run_root: Path = DEFAULT_RUN_ROOT,
    round_index: int | None = None,
    sample_index: int = 0,
    model: str = DEFAULT_MODEL,
    skill_file: Path | None = None,
) -> RunSpec:
    split = load_split(split_manifest)
    root = Path(data_root).expanduser().resolve()
    split.verify_source(root)

    if phase == "explore":
        resolved_round = validate_round_index(0 if round_index is None else round_index)
        validate_sample_index(sample_index)
        temperature = EXPLORE_TEMPERATURE
    elif phase == "test":
        if round_index is not None or sample_index != 0:
            raise ValueError("Test is one deterministic run")
        resolved_round = None
        temperature = TEST_TEMPERATURE
    else:
        raise ValueError(f"Unknown phase: {phase}")

    resolved_skill = None
    skill_sha256 = None
    if skill_file is not None:
        resolved_skill = Path(skill_file).expanduser().resolve()
        if not resolved_skill.is_file() or not resolved_skill.read_text(encoding="utf-8").strip():
            raise ValueError(f"Skill file is missing or empty: {resolved_skill}")
        skill_sha256 = hashlib.sha256(resolved_skill.read_bytes()).hexdigest()
    if phase == "explore" and resolved_round == 0 and resolved_skill is not None:
        raise ValueError("Explore round 0 must run without a skill")
    if phase == "explore" and resolved_round and resolved_skill is None:
        raise ValueError(f"Explore round {resolved_round} requires the previous skill")

    arm = "skill" if resolved_skill else "baseline"
    run_name = (
        f"moscibench_datacope_explore_round_{resolved_round}_sample_{sample_index:02d}"
        if phase == "explore"
        else f"moscibench_datacope_test_{arm}"
    )
    base = Path(run_root).expanduser().resolve()
    workspace_base = base / "workspaces"
    return RunSpec(
        phase=phase,
        round_index=resolved_round,
        sample_index=sample_index,
        task_ids=split.task_ids(phase),
        run_name=run_name,
        data_root=root,
        workspace_base=workspace_base,
        workspace_root=workspace_base / run_name,
        log_dir=base / "benchmarks" / run_name,
        model=str(model).strip(),
        temperature=temperature,
        skill_file=resolved_skill,
        skill_sha256=skill_sha256,
        split=split,
    )


def _score_manifest(spec: RunSpec, results_path: Path) -> dict[str, Any]:
    scores: dict[str, float] = {}
    with Path(results_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row["competition_id"])
            scores[task_id] = float(row["score"] or 0.0)
    if set(scores) != set(spec.task_ids):
        raise ValueError("DSLighting results differ from the frozen task set")
    datasets = {}
    for family in {task_family(task_id) for task_id in spec.task_ids}:
        values = [score for task_id, score in scores.items() if task_family(task_id) == family]
        datasets[family] = sum(values) / len(values)
    return {
        "phase": spec.phase,
        "round_index": spec.round_index,
        "sample_index": spec.sample_index,
        "task_ids": list(spec.task_ids),
        "model": spec.model,
        "temperature": spec.temperature,
        "skill_sha256": spec.skill_sha256,
        "accuracy": sum(scores.values()) / len(scores),
        "datasets": dict(sorted(datasets.items())),
    }


def run_react(
    spec: RunSpec,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_concurrency: int = 10,
    llm_max_concurrency: int = 10,
    timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> int:
    if min(max_steps, max_concurrency, llm_max_concurrency, timeout_seconds) <= 0:
        raise ValueError("Runtime limits must be positive")
    for path in (spec.workspace_root, spec.log_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Experiment output must be empty or new: {path}")

    require_bubblewrap()
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    from dslighting import configure_logging
    from dslighting.api.benchmark import DSBenchmark
    from dslighting.benchmark.core.source_catalog import get_benchmark_source_catalog
    from dslighting.core.config.builder import ConfigBuilder
    from dslighting.core.config.llm_resolution import build_llm_config

    agent_runtime: dict[str, object] = {"max_steps": max_steps}
    if spec.skill_file is not None:
        agent_runtime["skill_path"] = str(spec.skill_file)
    llm_config = build_llm_config(
        model=spec.model,
        temperature=spec.temperature,
        max_concurrent_per_key=llm_max_concurrency,
        global_max_concurrency=llm_max_concurrency,
    )
    config = ConfigBuilder().build_config(
        workflow="react",
        llm_config=llm_config,
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
        data_analysis={"cache_enabled": False, "profile": "full"},
        agent_runtime=agent_runtime,
        output_contract={
            "require_output_before_completion": True,
            "missing_output_feedback_retries": 2,
        },
    )
    config.run.parameters.update(
        {
            "datacope_split_id": spec.split.split_id,
            "datacope_phase": spec.phase,
            "datacope_round_index": spec.round_index,
            "datacope_sample_index": spec.sample_index,
            "datacope_skill_sha256": spec.skill_sha256,
        }
    )
    config.run.dag_runtime.node_timeout_seconds = float(timeout_seconds)
    config.scheduler.max_concurrency = max_concurrency
    config.scheduler.gpu_policy = "cpu_default"
    config.scheduler.cpu_worker_pool_size = max_concurrency

    configure_logging(
        level="INFO",
        trace_llm=False,
        output_dir=str(spec.log_dir / "debug_logs"),
        force=True,
    )
    source = get_benchmark_source_catalog().get_source("moscibench")
    benchmark = DSBenchmark(
        benchmark_type="moscibench",
        exp_name=spec.run_name,
        data_dir=str(spec.data_root),
        vendor_comp_dir=str(source.registry_root),
        competitions=list(spec.task_ids),
    )
    print(
        f"phase={spec.phase} tasks={len(spec.task_ids)} temperature={spec.temperature} "
        f"skill={spec.skill_sha256 or 'none'}"
    )
    result = benchmark.run(config=config, log_path=str(spec.log_dir), verbose=True)
    manifest = _score_manifest(spec, Path(result.results_path))
    spec.workspace_root.mkdir(parents=True, exist_ok=True)
    (spec.workspace_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def summarize_test(run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    summary: dict[str, Any] = {}
    for arm in ("baseline", "skill"):
        path = root / "workspaces" / f"moscibench_datacope_test_{arm}" / "run_manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read completed {arm} run: {path}") from exc
        summary[arm] = {
            "accuracy": manifest["accuracy"],
            "task_count": len(manifest["task_ids"]),
            "datasets": manifest["datasets"],
        }
    summary["delta"] = float(summary["skill"]["accuracy"]) - float(summary["baseline"]["accuracy"])
    (root / "test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
