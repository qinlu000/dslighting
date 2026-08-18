"""Run isolated DataMind Python tasks through the FastPerception workflow."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from dslighting.config import LLMConfig
from dslighting.core.types import TaskDefinition
from dslighting.workflows.search.react.protocol import extract_answer_block

from .build_split import source_file as source_file_from_prompt

DEFAULT_SPLIT = Path(
    "/data/caoqinlu/projects/dslighting/artifacts/datamind_perception/train.parquet"
)
DEFAULT_SOURCE_ROOT = Path("/data/caoqinlu/datasets/DataMind-Data/rl/train_files")


@dataclass(frozen=True)
class DataMindPerceptionTask:
    task_id: str
    source_task_id: str
    question: str
    source_file: str
    ground_truth: str
    reward_style: str
    upstream_record: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class RunSettings:
    source_root: Path
    output_dir: Path
    workspace_dir: Path
    staging_dir: Path
    llm: LLMConfig
    perception_enabled: bool = True
    max_steps: int = 20
    max_history_chars: int = 48000
    keep_recent_turns: int = 14
    summary_trigger_turns: int = 18
    concurrency: int = 4
    timeout_seconds: int = 1800
    memory_mb: int = 8192
    cpu_cores: float = 4.0
    pids_limit: int = 256
    overwrite: bool = False
    retry_failed: bool = False


def _task_id(record: Mapping[str, Any]) -> str:
    extra = record.get("extra_info")
    if not isinstance(extra, Mapping):
        raise ValueError("DataMind row has no extra_info mapping")
    value = str(extra.get("index") or "").strip()
    if not value:
        raise ValueError("DataMind row has no extra_info.index")
    return value


def _question(record: Mapping[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt is None:
        prompt = []
    value = next(
        (
            str(message.get("content") or "").strip()
            for message in prompt
            if isinstance(message, Mapping) and message.get("role") == "user"
        ),
        "",
    )
    extra = record.get("extra_info")
    if not value and isinstance(extra, Mapping):
        value = str(extra.get("question") or "").strip()
    if not value:
        raise ValueError("DataMind row has no question")
    return value


def _ground_truth(record: Mapping[str, Any]) -> str:
    reward_model = record.get("reward_model")
    if not isinstance(reward_model, Mapping):
        return ""
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        return ""
    return str(ground_truth.get("ground_truth") or "").strip()


def _safe_component(task_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._-")[:80] or "task"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def load_tasks(path: Path, *, source_root: Path) -> list[DataMindPerceptionTask]:
    """Load only fields needed for execution; never expose source trajectories."""
    path = Path(path).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DataMind split does not exist: {path}")
    if not source_root.is_dir():
        raise ValueError(f"DataMind source directory does not exist: {source_root}")

    frame = pd.read_parquet(path)
    rows = [row.to_dict() for _, row in frame.iterrows()]
    source_task_ids = [_task_id(record) for record in rows]
    id_counts = Counter(source_task_ids)
    tasks: list[DataMindPerceptionTask] = []
    seen: set[str] = set()
    for row_index, (record, source_task_id) in enumerate(zip(rows, source_task_ids)):
        identifier = (
            source_task_id
            if id_counts[source_task_id] == 1
            else f"{source_task_id}__row_{row_index:06d}"
        )
        if identifier in seen:
            raise ValueError(f"Duplicate task id in {path}: {identifier}")
        seen.add(identifier)
        source_name = str(record.get("source_file") or "").strip()
        if not source_name:
            prompt = record.get("prompt")
            source_name = source_file_from_prompt([] if prompt is None else prompt)
        if Path(source_name).name != source_name:
            raise ValueError(f"Unsafe source filename for task {identifier!r}: {source_name!r}")
        if not (source_root / source_name).is_file():
            raise ValueError(
                f"Missing source file for row {row_index}, task {identifier!r}: {source_name}"
            )
        reward_model = record.get("reward_model")
        reward_style = (
            str(reward_model.get("style") or "").strip()
            if isinstance(reward_model, Mapping)
            else ""
        )
        tasks.append(
            DataMindPerceptionTask(
                task_id=identifier,
                source_task_id=source_task_id,
                question=_question(record),
                source_file=source_name,
                ground_truth=_ground_truth(record),
                reward_style=reward_style,
                upstream_record=record,
            )
        )
    if not tasks:
        raise ValueError(f"DataMind split is empty: {path}")
    return tasks


def select_tasks(
    tasks: Sequence[DataMindPerceptionTask],
    *,
    task_ids: Sequence[str] = (),
    index_expression: str | None = None,
    select_all: bool = False,
) -> list[DataMindPerceptionTask]:
    selectors = int(bool(task_ids)) + int(bool(index_expression)) + int(bool(select_all))
    if selectors != 1:
        raise ValueError("Choose exactly one of task_ids, index_expression, or select_all")
    if task_ids:
        by_id = {task.task_id: task for task in tasks}
        missing = [identifier for identifier in task_ids if identifier not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {missing}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task ids must be unique")
        return [by_id[identifier] for identifier in task_ids]
    if select_all:
        return list(tasks)

    expression = str(index_expression or "").strip()
    try:
        if "-" in expression:
            start_text, end_text = expression.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start or end > len(tasks):
                raise ValueError
            return list(tasks[start:end])
        indices = [int(value.strip()) for value in expression.split(",")]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError
        if any(index < 0 or index >= len(tasks) for index in indices):
            raise ValueError
        return [tasks[index] for index in indices]
    except ValueError as exc:
        raise ValueError(f"Invalid index expression {expression!r}; use '0-5' or '0,2,4'") from exc


def prepare_agent_visible_dir(
    task: DataMindPerceptionTask,
    *,
    source_root: Path,
    staging_root: Path,
) -> Path:
    """Stage exactly one source file so task-to-task data cannot leak."""
    source = (Path(source_root).expanduser().resolve() / task.source_file).resolve()
    source.relative_to(Path(source_root).expanduser().resolve())
    if not source.is_file():
        raise ValueError(f"Source file does not exist: {source}")
    stage_dir = Path(staging_root).expanduser().resolve() / _safe_component(task.task_id)
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / task.source_file
    if destination.is_symlink() and destination.resolve() != source:
        raise ValueError(f"Stale staging symlink points to a different source: {destination}")
    if destination.exists() and not destination.is_symlink():
        raise ValueError(f"Staging destination already exists and is not a symlink: {destination}")
    if not destination.exists():
        os.symlink(source, destination)
    return stage_dir


def build_task_definition(
    task: DataMindPerceptionTask,
    *,
    agent_visible_dir: Path,
    output_dir: Path,
) -> TaskDefinition:
    description = (
        "# Data Source\n"
        f"**The data source path is '{task.source_file}'.**\n\n"
        f"{task.question}"
    )
    execution_spec = {
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "task_type": "datasci",
        "description_text": description,
        "io_instructions": "",
        "agent_visible_dir": str(Path(agent_visible_dir).resolve()),
        "output_path": str(Path(output_dir).resolve() / "answer.txt"),
    }
    return TaskDefinition(
        task_id=task.task_id,
        task_type="datasci",
        mode="open_ended",
        payload={"execution_spec": execution_spec},
    )


def _last_task_record(
    records: Iterable[Mapping[str, Any]], task_id: str
) -> Mapping[str, Any] | None:
    matching = [record for record in records if str(record.get("task_id")) == task_id]
    return matching[-1] if matching else None


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_episode(
    task: DataMindPerceptionTask,
    *,
    output_dir: Path,
    result: Any,
    cost: float,
    usage: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(str(record.get("workspace_dir") or "")) if record else Path()
    artifacts = workspace_dir / "artifacts" if workspace_dir.name else Path()
    messages = _read_json(artifacts / "messages.json", []) if artifacts.name else []
    sessions = (
        _read_json(artifacts / "perception_sessions.json", []) if artifacts.name else []
    )
    answer = next(
        (
            extracted
            for message in reversed(messages)
            if isinstance(message, Mapping) and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and (extracted := extract_answer_block(message["content"])) is not None
        ),
        None,
    )
    runner_failed = isinstance(result, str) and result.startswith("[ERROR]")
    failed = runner_failed or answer is None
    payload = {
        "schema_version": 1,
        "episode_id": _safe_component(task.task_id),
        "task_id": task.task_id,
        "source_file": task.source_file,
        "question": task.question,
        "ground_truth": task.ground_truth,
        "reward_style": task.reward_style,
        "status": "failed" if failed else "completed",
        "answer": answer,
        "runner_result": _json_safe(result),
        "solver_messages": _json_safe(messages),
        "perception_sessions": _json_safe(sessions),
        "judge": None,
        "dslighting": {
            "cost": float(cost),
            "usage": _json_safe(dict(usage or {})),
            "workspace_dir": str(workspace_dir) if workspace_dir.name else "",
        },
    }
    (output_dir / "episode.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _existing_episode(output_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(output_dir / "episode.json", None)
    return payload if isinstance(payload, dict) else None


def _prepare_output_target(output_dir: Path, *, overwrite: bool, retry_failed: bool) -> str:
    existing = _existing_episode(output_dir)
    if existing is not None and not overwrite:
        if not retry_failed or existing.get("status") == "completed":
            return "skip"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not (overwrite or retry_failed):
            raise ValueError(f"Non-empty output has no reusable episode: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    return "run"


async def run_tasks(
    tasks: Sequence[DataMindPerceptionTask], settings: RunSettings
) -> dict[str, Any]:
    from dslighting.core.config.builder import ConfigBuilder
    from dslighting.runner import DSLightingRunner

    if not tasks:
        raise ValueError("At least one task must be selected")
    if settings.concurrency <= 0 or settings.max_steps <= 0:
        raise ValueError("concurrency and max_steps must be positive")
    if settings.summary_trigger_turns < settings.keep_recent_turns:
        raise ValueError("summary_trigger_turns must be >= keep_recent_turns")

    output_root = settings.output_dir.expanduser().resolve()
    workspace_root = settings.workspace_dir.expanduser().resolve()
    staging_root = settings.staging_dir.expanduser().resolve()
    for path in (output_root, workspace_root, staging_root):
        path.mkdir(parents=True, exist_ok=True)
    run_lock = (output_root / ".dslighting_run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        run_lock.close()
        raise RuntimeError(f"Run is already active for output directory: {output_root}") from exc

    config = ConfigBuilder().build_config(
        workflow="react",
        llm_config=settings.llm,
        workspace_dir=str(workspace_root),
        run_name=output_root.name,
        keep_workspace=True,
        keep_workspace_on_failure=True,
        sandbox={
            "backend": "local",
            "timeout": settings.timeout_seconds,
            "memory_mb": settings.memory_mb,
            "cpu_cores": settings.cpu_cores,
            "pids_limit": settings.pids_limit,
            "environment_policy": "allowlist",
            "network_policy": "disabled",
            "local_isolation": "bubblewrap",
        },
        data_analysis={"cache_enabled": False},
        agent_runtime={
            "max_steps": settings.max_steps,
            "perception_enabled": settings.perception_enabled,
            "context": {
                "max_history_chars": settings.max_history_chars,
                "keep_recent_turns": settings.keep_recent_turns,
                "summary_trigger_turns": settings.summary_trigger_turns,
            },
        },
    )
    runner = DSLightingRunner(config)
    evaluate_task = runner.get_eval_function()

    async def run_one(task: DataMindPerceptionTask) -> dict[str, Any]:
        task_output = output_root / _safe_component(task.task_id)
        disposition = _prepare_output_target(
            task_output,
            overwrite=settings.overwrite,
            retry_failed=settings.retry_failed,
        )
        if disposition == "skip":
            return {"task_id": task.task_id, "status": "skipped"}
        visible_dir = prepare_agent_visible_dir(
            task, source_root=settings.source_root, staging_root=staging_root
        )
        definition = build_task_definition(
            task, agent_visible_dir=visible_dir, output_dir=task_output
        )
        result, cost, usage = await evaluate_task(definition)
        record = _last_task_record(runner.get_run_records(), task.task_id)
        episode = write_episode(
            task,
            output_dir=task_output,
            result=result,
            cost=cost,
            usage=usage,
            record=record,
        )
        return {
            "task_id": task.task_id,
            "status": episode["status"],
            "episode": str(task_output / "episode.json"),
            "perception_sessions": len(episode["perception_sessions"]),
            "cost": float(cost),
        }

    semaphore = asyncio.Semaphore(settings.concurrency)

    async def run_indexed(index: int, task: DataMindPerceptionTask):
        async with semaphore:
            return index, await run_one(task)

    try:
        indexed = await asyncio.gather(
            *(run_indexed(index, task) for index, task in enumerate(tasks))
        )
    finally:
        run_lock.close()
    results = [result for _, result in sorted(indexed)]
    summary = {
        "workflow": "fastperception" if settings.perception_enabled else "react",
        "solver_model": config.llm.model,
        "perception_model": config.llm.model,
        "perception_enabled": settings.perception_enabled,
        "sandbox": "local/bubblewrap/offline",
        "task_count": len(tasks),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "total_cost": sum(float(item.get("cost") or 0.0) for item in results),
        "tasks": results,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "DEFAULT_SOURCE_ROOT",
    "DEFAULT_SPLIT",
    "DataMindPerceptionTask",
    "RunSettings",
    "build_task_definition",
    "load_tasks",
    "prepare_agent_visible_dir",
    "run_tasks",
    "select_tasks",
    "write_episode",
]
