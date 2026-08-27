"""Small execution helpers shared by the research benchmarks.

Benchmark-specific task loading, staging, prompting, and grading stay in each
benchmark module.  This file only owns the mechanical parts of a rollout.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import shutil
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, TypeVar

from dslighting.config import LLMConfig

Task = TypeVar("Task")
Result = TypeVar("Result")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def prepare_task_output(
    output_dir: Path,
    *,
    overwrite: bool,
    retry_failed: bool,
    result_file: str = "result.json",
    complete_field: str = "status",
    complete_value: Any = "completed",
    create: bool = True,
) -> dict[str, Any] | None:
    """Return a reusable result, otherwise prepare an empty output directory."""

    previous = existing_result(output_dir / result_file)
    if previous is not None and not overwrite:
        if not retry_failed or previous.get(complete_field) == complete_value:
            return previous
    if output_dir.exists() and any(output_dir.iterdir()):
        if not (overwrite or retry_failed):
            raise ValueError(
                f"Non-empty task output is not reusable: {output_dir}; "
                "use --overwrite or --retry-failed"
            )
        shutil.rmtree(output_dir)
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return None


def last_task_record(
    records: Iterable[Mapping[str, Any]], task_id: str
) -> Mapping[str, Any] | None:
    matches = [record for record in records if str(record.get("task_id")) == task_id]
    return matches[-1] if matches else None


def acquire_run_lock(output_dir: Path) -> TextIO:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".dslighting_run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Run is already active for output directory: {output_dir}") from exc
    return handle


async def run_batch(
    tasks: Sequence[Task],
    run_one: Callable[[Task], Awaitable[Result]],
    *,
    concurrency: int,
) -> list[Result]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(task: Task) -> Result:
        async with semaphore:
            return await run_one(task)

    return list(await asyncio.gather(*(guarded(task) for task in tasks)))


def cpu_thread_env(threads: int, *, tensorflow: bool = False) -> dict[str, str]:
    value = str(max(1, int(threads)))
    env = {
        name: value
        for name in (
            "BLIS_NUM_THREADS",
            "LOKY_MAX_CPU_COUNT",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    if tensorflow:
        env.update(TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS=value)
    return env


@dataclass(frozen=True)
class RuntimeConfig:
    workflow: str
    llm: LLMConfig
    workspace_dir: Path
    run_name: str
    sandbox: Mapping[str, Any]
    agent_runtime: Mapping[str, Any]
    cpu_threads: int
    tensorflow: bool = False
    output_contract: Mapping[str, Any] = field(
        default_factory=lambda: {
            "require_output_before_completion": True,
            "missing_output_feedback_retries": 2,
        }
    )


def create_runner(runtime: RuntimeConfig):
    from dslighting.core.config.builder import ConfigBuilder
    from dslighting.runner import DSLightingRunner

    config = ConfigBuilder().build_config(
        workflow=runtime.workflow,
        llm_config=runtime.llm,
        workspace_dir=str(runtime.workspace_dir),
        run_name=runtime.run_name,
        keep_workspace=True,
        keep_workspace_on_failure=True,
        sandbox=dict(runtime.sandbox),
        data_analysis={"cache_enabled": False},
        agent_runtime=dict(runtime.agent_runtime),
        output_contract=dict(runtime.output_contract),
    )
    config.run.parameters = {
        **dict(config.run.parameters or {}),
        "sandbox_env": cpu_thread_env(
            runtime.cpu_threads,
            tensorflow=runtime.tensorflow,
        ),
    }
    return DSLightingRunner(config)


__all__ = [
    "RuntimeConfig",
    "acquire_run_lock",
    "cpu_thread_env",
    "create_runner",
    "existing_result",
    "last_task_record",
    "prepare_task_output",
    "read_json",
    "read_jsonl",
    "run_batch",
    "sha256_file",
    "write_json",
    "write_jsonl",
]
