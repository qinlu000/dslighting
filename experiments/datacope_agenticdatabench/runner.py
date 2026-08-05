"""Run a frozen AgenticDataBench split with the existing DSLighting PoC."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dslighting.core.config.llm_resolution import build_llm_config
from experiments.agenticdatabench_poc.runner import (
    AgenticDataBenchTask,
    RunSettings,
    load_tasks,
    run_tasks,
)

from .protocol import (
    EXPLORE_TEMPERATURE,
    TEST_TEMPERATURE,
    validate_round_index,
    validate_sample_index,
)
from .split import (
    DEFAULT_DATASETS_RELATIVE_PATH,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_TASKS_RELATIVE_PATH,
    AgenticDataBenchSplit,
    load_split,
)

DEFAULT_RUN_ROOT = Path(__file__).with_name("runs")
DEFAULT_MODEL = "openai/DeepSeek-V4-Flash"
DEFAULT_MAX_STEPS = 30
DEFAULT_CONCURRENCY = 50
DEFAULT_TASK_TIMEOUT_SECONDS = 2 * 60 * 60


@dataclass(frozen=True)
class RunSpec:
    phase: str
    round_index: int | None
    sample_index: int
    task_ids: tuple[str, ...]
    run_name: str
    benchmark_root: Path
    tasks_file: Path
    dataset_root: Path
    output_dir: Path
    workspace_dir: Path
    staging_dir: Path
    model: str
    temperature: float
    skill_file: Path | None
    skill_sha256: str | None
    split: AgenticDataBenchSplit


def resolve_run_spec(
    *,
    phase: str,
    benchmark_root: Path,
    split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
    run_root: Path = DEFAULT_RUN_ROOT,
    tasks_file: Path | None = None,
    dataset_root: Path | None = None,
    round_index: int | None = None,
    sample_index: int = 0,
    model: str = DEFAULT_MODEL,
    skill_file: Path | None = None,
) -> RunSpec:
    benchmark = Path(benchmark_root).expanduser().resolve()
    tasks = (
        Path(tasks_file).expanduser().resolve()
        if tasks_file is not None
        else benchmark / DEFAULT_TASKS_RELATIVE_PATH
    )
    datasets = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else benchmark / DEFAULT_DATASETS_RELATIVE_PATH
    )
    if not benchmark.is_dir() or not datasets.is_dir():
        raise ValueError(f"Incomplete AgenticDataBench checkout: {benchmark}")
    split = load_split(split_manifest)
    split.verify_source(tasks)

    if phase == "explore":
        resolved_round = validate_round_index(0 if round_index is None else round_index)
        validate_sample_index(sample_index)
        temperature = EXPLORE_TEMPERATURE
    elif phase == "test":
        if round_index is not None or sample_index != 0:
            raise ValueError("Test is one deterministic run per arm")
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
        f"agenticdatabench_datacope_explore_round_{resolved_round}_sample_{sample_index:02d}"
        if phase == "explore"
        else f"agenticdatabench_datacope_test_{arm}"
    )
    root = Path(run_root).expanduser().resolve()
    return RunSpec(
        phase=phase,
        round_index=resolved_round,
        sample_index=sample_index,
        task_ids=split.task_ids(phase),
        run_name=run_name,
        benchmark_root=benchmark,
        tasks_file=tasks,
        dataset_root=datasets,
        output_dir=root / "solver_outputs" / run_name,
        workspace_dir=root / "workspaces" / run_name,
        staging_dir=root / "staging" / run_name,
        model=str(model).strip(),
        temperature=temperature,
        skill_file=resolved_skill,
        skill_sha256=skill_sha256,
        split=split,
    )


def _selected_tasks(spec: RunSpec) -> list[AgenticDataBenchTask]:
    by_id = {task.task_id: task for task in load_tasks(spec.tasks_file)}
    missing = [task_id for task_id in spec.task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"Frozen AgenticDataBench tasks are missing: {missing[:5]}")
    return [by_id[task_id] for task_id in spec.task_ids]


def run_react(
    spec: RunSpec,
    *,
    sandbox_backend: str = "docker",
    docker_image: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> int:
    if min(max_steps, concurrency, timeout_seconds) <= 0:
        raise ValueError("Runtime limits must be positive")
    if sandbox_backend == "docker" and not str(docker_image or "").strip():
        raise ValueError("AgenticDataBench Docker runs require a docker image")
    for path in (spec.output_dir, spec.workspace_dir, spec.staging_dir):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Experiment output must be empty or new: {path}")

    settings = RunSettings(
        benchmark_root=spec.benchmark_root,
        dataset_root=spec.dataset_root,
        output_dir=spec.output_dir,
        workspace_dir=spec.workspace_dir,
        staging_dir=spec.staging_dir,
        workflow="react",
        llm=build_llm_config(
            model=spec.model,
            temperature=spec.temperature,
            max_retries=30,
            max_concurrent_per_key=concurrency,
            global_max_concurrency=concurrency,
        ),
        max_steps=max_steps,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        sandbox_backend=sandbox_backend,
        docker_image=docker_image,
        local_isolation="bubblewrap",
        disable_network=True,
        skill_file=spec.skill_file,
    )
    summary = asyncio.run(run_tasks(_selected_tasks(spec), settings))
    manifest = {
        "phase": spec.phase,
        "round_index": spec.round_index,
        "sample_index": spec.sample_index,
        "task_ids": list(spec.task_ids),
        "model": spec.model,
        "temperature": spec.temperature,
        "skill_sha256": spec.skill_sha256,
        "completed": summary["completed"],
        "failed": summary["failed"],
    }
    (spec.output_dir / "datacope_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def evaluate_official(
    *,
    benchmark_root: Path,
    output_dir: Path,
    python_executable: str = sys.executable,
    gold_dir: Path | None = None,
) -> int:
    benchmark = Path(benchmark_root).expanduser().resolve()
    testbed = benchmark / "testbed"
    evaluator = testbed / "evaluate.py"
    output = Path(output_dir).expanduser().resolve()
    selected_tasks = output / "agenticdatabench_tasks.jsonl"
    gold = (
        Path(gold_dir).expanduser().resolve()
        if gold_dir is not None
        else testbed / "gold"
    )
    for path, label in (
        (evaluator, "official evaluator"),
        (selected_tasks, "selected task JSONL"),
        (gold, "gold directory"),
    ):
        if not path.exists():
            raise ValueError(f"AgenticDataBench {label} does not exist: {path}")
    completed = subprocess.run(
        [
            python_executable,
            str(evaluator),
            "--output_dir",
            str(output),
            "--gold_dir",
            str(gold),
            "--eval_json",
            str(selected_tasks),
        ],
        cwd=testbed,
        check=False,
    )
    return completed.returncode
