"""Run the deterministic DataMind SQL subset through DSLighting.

Only the SQLite database is exposed to the agent. DataMind retains ownership
of the questions and CSV ground truth; this module reproduces its programmatic
table-matching reward without importing the upstream model client, SQL
interpreter, or training stack.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from numbers import Number
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any, Mapping, Sequence

import pandas as pd

from dslighting.config import LLMConfig
from dslighting.core.types import TaskDefinition
from experiments.common import (
    RuntimeConfig,
    acquire_run_lock,
    create_runner,
    last_task_record,
    prepare_task_output,
    run_batch,
    write_json,
    write_jsonl,
)

RESULT_FILENAME = "result.csv"
SELECTED_TASKS_FILENAME = "datamind_sql_tasks.jsonl"


def _safe_component(value: str, *, field_name: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be one safe path component: {value!r}")
    return value


def _safe_relative_path(value: str, *, field_name: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise ValueError(f"{field_name} must be a safe relative path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class DataMindSQLTask:
    """One validated SQL task from the canonical training-mixture manifest."""

    task_id: str
    db_id: str
    question: str
    database_relpath: str
    gold_relpath: str
    ignore_order: bool = True
    upstream_record: Mapping[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DataMindSQLTask":
        if payload.get("source") != "datamind_sql":
            raise ValueError("Expected a datamind_sql manifest record")
        task_id = _safe_component(str(payload.get("task_id") or "").strip(), field_name="task_id")
        db_id = _safe_component(
            str(payload.get("dataset_key") or "").strip(), field_name="dataset_key"
        )
        question = str(payload.get("question") or "").strip()
        input_ref = payload.get("input_ref")
        evaluator = payload.get("evaluator")
        if not isinstance(input_ref, Mapping) or input_ref.get("kind") != "sqlite":
            raise ValueError(f"Task {task_id} must declare a SQLite input")
        if not isinstance(evaluator, Mapping) or evaluator.get("kind") != "csv_table_match":
            raise ValueError(f"Task {task_id} must declare csv_table_match")
        if evaluator.get("agent_visible") is not False:
            raise ValueError(f"Task {task_id} must keep ground truth evaluator-only")
        if not question:
            raise ValueError(f"Task {task_id} has an empty question")

        database_relpath = _safe_relative_path(
            str(input_ref.get("ref") or "").strip(), field_name=f"{task_id}.input_ref"
        )
        gold_relpath = _safe_relative_path(
            str(evaluator.get("ground_truth_ref") or "").strip(),
            field_name=f"{task_id}.ground_truth_ref",
        )
        expected_database = f"rl/train_files/{db_id}.sqlite"
        expected_gold = f"rl/gold_csv_results/{task_id}.csv"
        if database_relpath != expected_database or gold_relpath != expected_gold:
            raise ValueError(f"Task {task_id} has inconsistent DataMind artifact paths")
        return cls(
            task_id=task_id,
            db_id=db_id,
            question=question,
            database_relpath=database_relpath,
            gold_relpath=gold_relpath,
            ignore_order=bool(evaluator.get("ignore_order", False)),
            upstream_record=dict(payload),
        )

    def manifest_record(self) -> Mapping[str, Any]:
        return self.upstream_record or {
            "source": "datamind_sql",
            "task_id": self.task_id,
            "dataset_key": self.db_id,
            "question": self.question,
            "input_ref": {"kind": "sqlite", "ref": self.database_relpath},
            "evaluator": {
                "kind": "csv_table_match",
                "ignore_order": self.ignore_order,
                "ground_truth_ref": self.gold_relpath,
                "agent_visible": False,
            },
        }


def load_tasks(path: Path) -> list[DataMindSQLTask]:
    """Load only DataMind SQL rows from a mixed JSONL manifest."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Training-mixture manifest does not exist: {path}")
    tasks: list[DataMindSQLTask] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            if payload.get("source") != "datamind_sql":
                continue
            task = DataMindSQLTask.from_payload(payload)
            if task.task_id in seen:
                raise ValueError(f"Duplicate DataMind SQL task id: {task.task_id}")
            seen.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"Manifest contains no DataMind SQL tasks: {path}")
    return tasks


def select_tasks(
    tasks: Sequence[DataMindSQLTask],
    *,
    task_ids: Sequence[str] = (),
    index_expression: str | None = None,
    select_all: bool = False,
) -> list[DataMindSQLTask]:
    if int(bool(task_ids)) + int(bool(index_expression)) + int(bool(select_all)) != 1:
        raise ValueError("Choose exactly one of task_ids, index_expression, or select_all")
    if task_ids:
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {missing}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task ids must be unique")
        return [by_id[task_id] for task_id in task_ids]
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


@dataclass(frozen=True)
class DataMindSQLDataStore:
    dataset_root: Path

    def __post_init__(self) -> None:
        root = self.dataset_root.expanduser().resolve()
        if (
            not (root / "rl" / "train_files").is_dir()
            or not (root / "rl" / "gold_csv_results").is_dir()
        ):
            raise ValueError(f"Invalid DataMind dataset root: {root}")
        object.__setattr__(self, "dataset_root", root)

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.dataset_root / relative_path).resolve()
        try:
            candidate.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError(f"DataMind path escapes dataset root: {relative_path}") from exc
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise ValueError(f"Missing or empty DataMind artifact: {candidate}")
        return candidate

    def stage_inputs(self, task: DataMindSQLTask, staging_root: Path) -> Path:
        database = self.resolve(task.database_relpath)
        stage_dir = Path(staging_root).expanduser().resolve() / task.task_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        link = stage_dir / "database.sqlite"
        if link.is_symlink():
            if link.resolve() != database:
                raise ValueError(f"Staged database points to the wrong file: {link}")
        elif link.exists():
            raise ValueError(f"Staged database is not a symlink: {link}")
        else:
            os.symlink(database, link)
        return stage_dir

    def read_gold(self, task: DataMindSQLTask) -> pd.DataFrame:
        return pd.read_csv(self.resolve(task.gold_relpath))


def build_task_definition(
    task: DataMindSQLTask,
    *,
    agent_visible_dir: Path,
    output_path: Path,
) -> TaskDefinition:
    output_path = Path(output_path).expanduser().resolve()
    io_instructions = (
        "The read-only SQLite database is `database.sqlite` in the current working directory. "
        "Inspect it with Python's sqlite3 module and answer the question by querying the database. "
        "Write the resulting table to `result.csv` in the current working directory with no index "
        "column. Each Action may run in a fresh process, so persist results in files. Before "
        "finishing, verify that `result.csv` exists and contains the requested rows and columns."
    )
    execution_spec = {
        "task_id": task.task_id,
        "task_type": "datasci",
        "description_text": task.question,
        "io_instructions": io_instructions,
        "agent_visible_dir": str(Path(agent_visible_dir).expanduser().resolve()),
        "output_path": str(output_path),
        "metric_name": "binary_reward",
        "lower_is_better": False,
        "source_id": "datamind-sql",
        "engine_id": "datamind-sql",
        "submission_artifact_contract": {
            "output_submission_path": str(output_path),
            "submission_filename": RESULT_FILENAME,
            "submission_format": ".csv",
            "root_kind": "file",
            "root_basename": "result",
            "entries": [
                {
                    "relative_path": RESULT_FILENAME,
                    "format": "csv",
                    "description": "SQL query result table",
                }
            ],
        },
    }
    return TaskDefinition(
        task_id=task.task_id,
        task_type="datasci",
        payload={"execution_spec": execution_spec},
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _vectors_match(left: list[Any], right: list[Any], *, ignore_order: bool) -> bool:
    if ignore_order:

        def key(value: Any) -> tuple[bool, str, bool]:
            return value is None, str(value), _is_number(value)

        left, right = sorted(left, key=key), sorted(right, key=key)
    if len(left) != len(right):
        return False
    for expected, predicted in zip(left, right):
        if pd.isna(expected) and pd.isna(predicted):
            continue
        if _is_number(expected) and _is_number(predicted):
            if not math.isclose(float(expected), float(predicted), abs_tol=1e-2):
                return False
        elif expected != predicted:
            return False
    return True


def tables_match(prediction: pd.DataFrame, gold: pd.DataFrame, *, ignore_order: bool) -> bool:
    """Reproduce DataMind's column-wise CSV comparison semantics."""

    predicted_columns = prediction.transpose().values.tolist()
    gold_columns = gold.transpose().values.tolist()
    return all(
        any(
            _vectors_match(expected, candidate, ignore_order=ignore_order)
            for candidate in predicted_columns
        )
        for expected in gold_columns
    )


def grade_result(
    task: DataMindSQLTask,
    result_path: Path,
    data_store: DataMindSQLDataStore,
) -> dict[str, Any]:
    result_path = Path(result_path).expanduser().resolve()
    if not result_path.is_file():
        return {"result_exist": 0.0, "reward": 0.0, "method": "missing_result"}
    try:
        prediction = pd.read_csv(result_path)
        gold = data_store.read_gold(task)
    except Exception as exc:
        return {
            "result_exist": 1.0,
            "reward": 0.0,
            "method": "unreadable_result",
            "detail": str(exc),
        }
    matched = tables_match(prediction, gold, ignore_order=task.ignore_order)
    return {
        "result_exist": 1.0,
        "reward": 1.0 if matched else 0.0,
        "method": "csv_table_match" if matched else "mismatch",
        "prediction_shape": list(prediction.shape),
        "gold_shape": list(gold.shape),
    }


@dataclass(frozen=True)
class RunSettings:
    data_store: DataMindSQLDataStore
    output_dir: Path
    workspace_dir: Path
    staging_dir: Path
    workflow: str
    llm: LLMConfig
    perception_enabled: bool = False
    perception_llm: LLMConfig | None = None
    max_steps: int = 10
    protocol_mode: str = "repair"
    max_history_chars: int = 48000
    keep_recent_turns: int = 14
    summary_trigger_turns: int = 18
    recent_observation_window: int = 8
    keep_latest_feedback_only: bool = True
    max_feedback_retries: int = 1
    concurrency: int = 50
    timeout_seconds: int = 600
    local_isolation: str = "bubblewrap"
    sandbox_python: Path | None = None
    compute_threads: int = 2
    overwrite: bool = False
    retry_failed: bool = False


async def run_tasks(tasks: Sequence[DataMindSQLTask], settings: RunSettings) -> dict[str, Any]:
    if not tasks:
        raise ValueError("At least one DataMind SQL task must be selected")
    if (
        min(
            settings.concurrency,
            settings.max_steps,
            settings.timeout_seconds,
            settings.compute_threads,
        )
        <= 0
    ):
        raise ValueError("Concurrency, steps, timeout, and compute threads must be positive")

    output_root = settings.output_dir.expanduser().resolve()
    workspace_root = settings.workspace_dir.expanduser().resolve()
    staging_root = settings.staging_dir.expanduser().resolve()
    for path in (output_root, workspace_root, staging_root):
        path.mkdir(parents=True, exist_ok=True)

    lock_handle = acquire_run_lock(output_root)

    selected_manifest = output_root / SELECTED_TASKS_FILENAME
    write_jsonl(selected_manifest, (task.manifest_record() for task in tasks))

    sandbox: dict[str, Any] = {
        "backend": "local",
        "timeout": settings.timeout_seconds,
        "memory_mb": 8192,
        "cpu_cores": float(settings.compute_threads),
        "pids_limit": 256,
        "environment_policy": "allowlist",
        "network_policy": "disabled",
        "local_isolation": settings.local_isolation,
    }
    if settings.sandbox_python is not None:
        sandbox["python_executable"] = str(settings.sandbox_python.expanduser().absolute())
    runner = create_runner(
        RuntimeConfig(
            workflow=settings.workflow,
            llm=settings.llm,
            workspace_dir=workspace_root,
            run_name=output_root.name,
            sandbox=sandbox,
            agent_runtime={
                "max_steps": settings.max_steps,
                "protocol_mode": settings.protocol_mode,
                "perception_enabled": settings.perception_enabled,
                **(
                    {"perception_llm": settings.perception_llm.model_dump()}
                    if settings.perception_llm is not None
                    else {}
                ),
                "context": {
                    "max_history_chars": settings.max_history_chars,
                    "keep_recent_turns": settings.keep_recent_turns,
                    "summary_trigger_turns": settings.summary_trigger_turns,
                    "recent_observation_window": settings.recent_observation_window,
                    "keep_latest_feedback_only": settings.keep_latest_feedback_only,
                    "max_feedback_retries": settings.max_feedback_retries,
                },
            },
            cpu_threads=settings.compute_threads,
        )
    )
    config = runner.config
    evaluate_task = runner.get_eval_function()

    async def run_one(task: DataMindSQLTask) -> dict[str, Any]:
        task_output = output_root / task.task_id
        existing = prepare_task_output(
            task_output,
            overwrite=settings.overwrite,
            retry_failed=settings.retry_failed,
        )
        if existing is not None:
            return {
                "task_id": task.task_id,
                "status": "skipped",
                "reward": float(existing.get("reward") or 0.0),
                "result": str(task_output / "result.json"),
            }
        result_path = task_output / RESULT_FILENAME
        try:
            visible_dir = settings.data_store.stage_inputs(task, staging_root)
            definition = build_task_definition(
                task, agent_visible_dir=visible_dir, output_path=result_path
            )
            result, cost, usage = await evaluate_task(definition)
            if isinstance(result, str) and result.startswith("[ERROR]"):
                raise RuntimeError(result)
            record = last_task_record(runner.get_run_records(), task.task_id)
            grade = grade_result(task, result_path, settings.data_store)
            status = "completed" if grade.get("result_exist") else "failed"
            payload = {
                "schema_version": 1,
                "task_id": task.task_id,
                "db_id": task.db_id,
                "status": status,
                "reward": float(grade["reward"]),
                "grade": grade,
                "result_path": str(result_path),
                "runner_result": str(result) if isinstance(result, Path) else result,
                "dslighting": {
                    "workflow": settings.workflow,
                    "model": settings.llm.model,
                    "cost": float(cost),
                    "usage": dict(usage or {}),
                    "workspace_dir": str(record.get("workspace_dir") or "") if record else "",
                },
            }
        except Exception as exc:
            status = "failed"
            payload = {
                "schema_version": 1,
                "task_id": task.task_id,
                "db_id": task.db_id,
                "status": status,
                "reward": 0.0,
                "grade": {
                    "result_exist": 0.0,
                    "reward": 0.0,
                    "method": "error",
                    "detail": str(exc),
                },
                "result_path": str(result_path),
                "dslighting": {
                    "workflow": settings.workflow,
                    "model": settings.llm.model,
                    "cost": 0.0,
                    "usage": {},
                    "workspace_dir": "",
                },
            }
        write_json(task_output / "result.json", payload)
        return {
            "task_id": task.task_id,
            "status": status,
            "reward": float(payload["reward"]),
            "cost": float(payload["dslighting"]["cost"]),
            "result": str(task_output / "result.json"),
        }

    try:
        results = await run_batch(
            tasks,
            run_one,
            concurrency=settings.concurrency,
        )
    finally:
        lock_handle.close()
    rewards = [float(item.get("reward") or 0.0) for item in results]
    summary = {
        "schema_version": 1,
        "benchmark": "datamind-sql-train",
        "workflow": settings.workflow,
        "model": config.llm.model,
        "thinking": config.llm.thinking,
        "sandbox": {
            "backend": "local",
            "local_isolation": settings.local_isolation,
            "python_executable": str(config.sandbox.python_executable or ""),
            "network_disabled": True,
        },
        "task_manifest": str(selected_manifest),
        "task_count": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "average_reward": fmean(rewards) if rewards else 0.0,
        "total_cost": sum(float(item.get("cost") or 0.0) for item in results),
        "tasks": results,
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


__all__ = [
    "DataMindSQLDataStore",
    "DataMindSQLTask",
    "RESULT_FILENAME",
    "RunSettings",
    "build_task_definition",
    "grade_result",
    "load_tasks",
    "run_tasks",
    "select_tasks",
    "tables_match",
]
