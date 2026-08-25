"""Run fixed-ground-truth DARE-Bench tasks through DSLighting workflows.

DARE-Bench retains ownership of task prompts, input files, and deterministic
grading. DSLighting supplies the solving workflow, workspace, sandbox, and
trajectory telemetry. Evaluator-only fields and ground truth are never staged
into the agent-visible directory.
"""

from __future__ import annotations

import asyncio
import fcntl
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

from dslighting.config import LLMConfig
from dslighting.core.types import TaskDefinition

PREDICTION_FILENAME = "prediction.csv"
SELECTED_TASKS_FILENAME = "dare_bench_tasks.jsonl"
SUPPORTED_TASK_TYPES = {"classification", "regression", "time_series_analysis"}
SUPPORTED_VERSIONS = {"v1", "v2"}
SUPPORTED_METRICS = {"macro_f1", "clipped_r2"}


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


@dataclass(frozen=True)
class DareBenchTask:
    """A validated fixed-GT DARE training sample."""

    task_id: str
    base_task_id: str
    version: str
    task_type: str
    question: str
    needed_files: tuple[str, ...]
    needed_file_members: tuple[str, ...]
    targets: tuple[str, ...]
    metric: str
    ground_truth_relpath: str
    prediction_filename: str = PREDICTION_FILENAME
    upstream_record: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @property
    def output_key(self) -> str:
        return f"{self.base_task_id}__{self.version}"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DareBenchTask":
        agent = payload.get("agent_payload")
        grader = payload.get("grader")
        metadata = payload.get("metadata")
        if not all(isinstance(value, Mapping) for value in (agent, grader, metadata)):
            raise ValueError("DARE record must contain agent_payload, grader, and metadata objects")

        task_id = str(payload.get("task_id") or "").strip()
        base_task_id = _safe_component(
            str(payload.get("base_task_id") or "").strip(), field_name="base_task_id"
        )
        version = str(payload.get("version") or "").strip()
        task_type = str(payload.get("task_type") or "").strip()
        question = str(agent.get("question") or "").strip()
        needed_files = tuple(str(value).strip() for value in agent.get("needed_files") or ())
        needed_members = tuple(
            str(value).strip() for value in agent.get("needed_file_members") or ()
        )
        targets_raw = metadata.get("target")
        targets = (
            (str(targets_raw).strip(),)
            if isinstance(targets_raw, str)
            else tuple(str(value).strip() for value in targets_raw or ())
        )
        metric = str(grader.get("metric") or "").strip()
        ground_truth_relpath = str(grader.get("ground_truth_relpath") or "").strip()
        prediction_filename = str(grader.get("prediction_filename") or PREDICTION_FILENAME).strip()

        if task_id != f"{base_task_id}::{version}":
            raise ValueError(f"Invalid versioned task id: {task_id!r}")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported DARE version for {task_id}: {version!r}")
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported DARE task type for {task_id}: {task_type!r}")
        if not question:
            raise ValueError(f"Task {task_id!r} has an empty question")
        if not needed_files or len(needed_files) != len(needed_members):
            raise ValueError(f"Task {task_id!r} has inconsistent needed-file lists")
        if len(set(needed_files)) != len(needed_files):
            raise ValueError(f"Task {task_id!r} has duplicate needed files")
        for name in needed_files:
            _safe_component(name, field_name=f"{task_id}.needed_files")
        expected_prefix = f"{base_task_id}/source/"
        for name, member in zip(needed_files, needed_members):
            if member != f"{expected_prefix}{name}":
                raise ValueError(f"Task {task_id!r} has an unsafe source member: {member!r}")
        if not targets or any(not target for target in targets):
            raise ValueError(f"Task {task_id!r} has no valid targets")
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric for {task_id}: {metric!r}")
        expected_gt_prefix = f"{base_task_id}/verify/"
        if not ground_truth_relpath.startswith(expected_gt_prefix):
            raise ValueError(f"Task {task_id!r} has an unsafe ground-truth path")
        if prediction_filename != PREDICTION_FILENAME:
            raise ValueError(f"Task {task_id!r} must produce {PREDICTION_FILENAME}")

        return cls(
            task_id=task_id,
            base_task_id=base_task_id,
            version=version,
            task_type=task_type,
            question=question,
            needed_files=needed_files,
            needed_file_members=needed_members,
            targets=targets,
            metric=metric,
            ground_truth_relpath=ground_truth_relpath,
            prediction_filename=prediction_filename,
            upstream_record=dict(payload),
        )


def load_tasks(path: Path) -> list[DareBenchTask]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DARE manifest does not exist: {path}")
    tasks: list[DareBenchTask] = []
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
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            task = DareBenchTask.from_payload(payload)
            if task.task_id in seen:
                raise ValueError(f"Duplicate task id in {path}: {task.task_id}")
            seen.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"DARE manifest is empty: {path}")
    return tasks


def select_tasks(
    tasks: Sequence[DareBenchTask],
    *,
    task_ids: Sequence[str] = (),
    index_expression: str | None = None,
    select_all: bool = False,
) -> list[DareBenchTask]:
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
class DareDataStore:
    """Read DARE artifacts from either databases.zip or an extracted database root."""

    databases_zip: Path | None = None
    databases_dir: Path | None = None

    def __post_init__(self) -> None:
        zip_path = self.databases_zip.expanduser().resolve() if self.databases_zip else None
        directory = self.databases_dir.expanduser().resolve() if self.databases_dir else None
        if (zip_path is None) == (directory is None):
            raise ValueError("Provide exactly one of databases_zip or databases_dir")
        if zip_path is not None and not zip_path.is_file():
            raise ValueError(f"DARE databases archive does not exist: {zip_path}")
        if directory is not None and not directory.is_dir():
            raise ValueError(f"DARE databases directory does not exist: {directory}")
        object.__setattr__(self, "databases_zip", zip_path)
        object.__setattr__(self, "databases_dir", directory)

    def read_bytes(self, relative_path: str) -> bytes:
        if self.databases_zip is not None:
            try:
                with zipfile.ZipFile(self.databases_zip) as archive:
                    return archive.read(relative_path)
            except KeyError as exc:
                raise ValueError(f"Missing DARE archive member: {relative_path}") from exc
        assert self.databases_dir is not None
        candidate = (self.databases_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.databases_dir)
        except ValueError as exc:
            raise ValueError(f"DARE path escapes database root: {relative_path}") from exc
        if not candidate.is_file():
            raise ValueError(f"Missing DARE file: {candidate}")
        return candidate.read_bytes()

    def stage_inputs(self, task: DareBenchTask, staging_root: Path) -> Path:
        stage_dir = Path(staging_root).expanduser().resolve() / task.output_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        for name, member in zip(task.needed_files, task.needed_file_members):
            destination = stage_dir / name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            payload = self.read_bytes(member)
            if not payload:
                raise ValueError(f"Empty DARE input file: {member}")
            fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=stage_dir)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                os.replace(temporary_name, destination)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        return stage_dir

    def read_ground_truth(self, task: DareBenchTask) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(task.ground_truth_relpath)))


def build_task_definition(
    task: DareBenchTask,
    *,
    agent_visible_dir: Path,
    output_path: Path,
) -> TaskDefinition:
    output_path = Path(output_path).expanduser().resolve()
    execution_spec = {
        "task_id": task.task_id,
        "task_type": "datasci",
        "description_text": task.question,
        "io_instructions": "",
        "agent_visible_dir": str(Path(agent_visible_dir).expanduser().resolve()),
        "output_path": str(output_path),
        "metric_name": task.metric,
        "lower_is_better": False,
        "source_id": "dare-bench",
        "engine_id": "dare",
        "submission_artifact_contract": {
            "output_submission_path": str(output_path),
            "submission_filename": PREDICTION_FILENAME,
            "submission_format": ".csv",
            "root_kind": "file",
            "root_basename": "prediction",
            "entries": [
                {
                    "relative_path": PREDICTION_FILENAME,
                    "format": "csv",
                    "description": "DARE-Bench prediction artifact",
                }
            ],
        },
    }
    return TaskDefinition(
        task_id=task.task_id,
        task_type="datasci",
        payload={"execution_spec": execution_spec},
    )


def _alignment_columns(task: DareBenchTask, ground_truth: pd.DataFrame) -> list[str]:
    target_set = set(task.targets)
    if task.task_type == "time_series_analysis" and task.version == "v2":
        return [column for column in ground_truth.columns if column not in target_set]
    if "row_id" in ground_truth.columns:
        return ["row_id"]
    return [column for column in ground_truth.columns if column not in target_set]


def _score_column(task: DareBenchTask, predicted: pd.Series, expected: pd.Series) -> float:
    try:
        if task.task_type == "classification":
            return float(f1_score(expected.tolist(), predicted.tolist(), average="macro"))
        return float(np.clip(r2_score(expected.tolist(), predicted.tolist()), 0.0, 1.0))
    except Exception:
        return 0.0


def score_prediction(
    task: DareBenchTask,
    prediction_path: Path,
    data_store: DareDataStore,
) -> dict[str, Any]:
    """Apply the same alignment and metrics as DARE-Bench evaluation.py."""

    prediction_path = Path(prediction_path).expanduser().resolve()
    if not prediction_path.is_file():
        return {"prediction_exist": 0.0, "final_score": 0.0, "error": "missing_prediction"}
    try:
        ground_truth = data_store.read_ground_truth(task)
        prediction = pd.read_csv(prediction_path)
    except Exception as exc:
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "unreadable_artifact",
            "detail": str(exc),
        }

    id_columns = _alignment_columns(task, ground_truth)
    if not id_columns:
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "empty_id_columns_after_resolving_alignment",
            "id_columns": [],
        }
    missing_ids = [column for column in id_columns if column not in prediction.columns]
    if missing_ids:
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "prediction_missing_id_columns",
            "id_columns": id_columns,
            "missing_id_columns": missing_ids,
        }

    prediction_columns = id_columns + [
        column for column in task.targets if column in prediction.columns
    ]
    try:
        merged = ground_truth.merge(
            prediction[prediction_columns],
            on=id_columns,
            how="inner",
            suffixes=("_gt", "_pred"),
        )
    except Exception as exc:
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "prediction_merge_failed",
            "detail": str(exc),
            "id_columns": id_columns,
        }
    if len(merged) != len(ground_truth):
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "row_count_mismatch_after_merge",
            "id_columns": id_columns,
            "ground_truth_rows": len(ground_truth),
            "merged_rows": len(merged),
        }

    per_target: dict[str, float] = {}
    for column in task.targets:
        if column not in prediction.columns:
            per_target[column] = 0.0
        else:
            per_target[column] = _score_column(
                task,
                merged[f"{column}_pred"],
                merged[f"{column}_gt"],
            )
    return {
        "prediction_exist": 1.0,
        "final_score": float(np.mean(list(per_target.values()))),
        "per_target_scores": per_target,
        "id_columns": id_columns,
        "ground_truth_rows": len(ground_truth),
    }


@dataclass(frozen=True)
class RunSettings:
    data_store: DareDataStore
    output_dir: Path
    workspace_dir: Path
    staging_dir: Path
    workflow: str
    llm: LLMConfig
    max_steps: int = 10
    max_history_chars: int = 48000
    keep_recent_turns: int = 14
    summary_trigger_turns: int = 18
    concurrency: int = 30
    timeout_seconds: int = 3600
    sandbox_backend: str = "local"
    docker_image: str | None = None
    memory_mb: int = 8192
    cpu_cores: float = 4.0
    pids_limit: int = 256
    local_isolation: str = "bubblewrap"
    sandbox_python: Path | None = None
    disable_network: bool = True
    overwrite: bool = False
    retry_failed: bool = False


def _last_task_record(
    records: Iterable[Mapping[str, Any]], task_id: str
) -> Mapping[str, Any] | None:
    matches = [record for record in records if str(record.get("task_id")) == task_id]
    return matches[-1] if matches else None


def _existing_result(task_output: Path) -> dict[str, Any] | None:
    result_path = task_output / "result.json"
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _prepare_output_target(task_output: Path, *, overwrite: bool, retry_failed: bool) -> str:
    existing = _existing_result(task_output)
    if existing is not None and not overwrite:
        if not retry_failed or existing.get("status") == "completed":
            return "skip"
    if task_output.exists() and any(task_output.iterdir()):
        if not (overwrite or retry_failed):
            raise ValueError(
                f"Non-empty task output has no reusable result: {task_output}; "
                "use --overwrite or --retry-failed"
            )
        shutil.rmtree(task_output)
    task_output.mkdir(parents=True, exist_ok=True)
    return "run"


async def run_tasks(tasks: Sequence[DareBenchTask], settings: RunSettings) -> dict[str, Any]:
    from dslighting.core.config.builder import ConfigBuilder
    from dslighting.runner import DSLightingRunner

    if not tasks:
        raise ValueError("At least one DARE task must be selected")
    if settings.concurrency <= 0 or settings.max_steps <= 0:
        raise ValueError("concurrency and max_steps must be positive")
    if settings.summary_trigger_turns < settings.keep_recent_turns:
        raise ValueError("summary_trigger_turns must be >= keep_recent_turns")
    if settings.sandbox_backend == "docker" and not str(settings.docker_image or "").strip():
        raise ValueError("Docker sandbox requires docker_image")

    output_root = settings.output_dir.expanduser().resolve()
    workspace_root = settings.workspace_dir.expanduser().resolve()
    staging_root = settings.staging_dir.expanduser().resolve()
    for path in (output_root, workspace_root, staging_root):
        path.mkdir(parents=True, exist_ok=True)

    lock_handle = (output_root / ".dslighting_run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Run is already active for output directory: {output_root}") from exc

    selected_manifest = output_root / SELECTED_TASKS_FILENAME
    selected_manifest.write_text(
        "".join(
            json.dumps(task.upstream_record, ensure_ascii=False, sort_keys=True) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )

    sandbox: dict[str, Any] = {
        "backend": settings.sandbox_backend,
        "timeout": settings.timeout_seconds,
        "memory_mb": settings.memory_mb,
        "cpu_cores": settings.cpu_cores,
        "pids_limit": settings.pids_limit,
    }
    if settings.sandbox_backend in {"local", "docker"}:
        sandbox.update(
            {
                "environment_policy": "allowlist",
                "network_policy": "disabled" if settings.disable_network else "inherit",
            }
        )
    if settings.sandbox_backend == "local":
        sandbox["local_isolation"] = settings.local_isolation
        if settings.sandbox_python is not None:
            sandbox["python_executable"] = str(settings.sandbox_python.expanduser().absolute())
    elif settings.sandbox_backend == "docker":
        sandbox["docker_image"] = str(settings.docker_image).strip()

    config = ConfigBuilder().build_config(
        workflow=settings.workflow,
        llm_config=settings.llm,
        workspace_dir=str(workspace_root),
        run_name=output_root.name,
        keep_workspace=True,
        keep_workspace_on_failure=True,
        sandbox=sandbox,
        data_analysis={"cache_enabled": False},
        agent_runtime={
            "max_steps": settings.max_steps,
            "context": {
                "max_history_chars": settings.max_history_chars,
                "keep_recent_turns": settings.keep_recent_turns,
                "summary_trigger_turns": settings.summary_trigger_turns,
            },
        },
        output_contract={
            "require_output_before_completion": True,
            "missing_output_feedback_retries": 2,
        },
    )
    runner = DSLightingRunner(config)
    evaluate_task = runner.get_eval_function()

    async def run_one(task: DareBenchTask) -> dict[str, Any]:
        task_output = output_root / task.output_key
        disposition = _prepare_output_target(
            task_output,
            overwrite=settings.overwrite,
            retry_failed=settings.retry_failed,
        )
        if disposition == "skip":
            existing = _existing_result(task_output) or {}
            return {
                "task_id": task.task_id,
                "status": "skipped",
                "score": float(existing.get("score") or 0.0),
                "result": str(task_output / "result.json"),
            }

        print(
            f"run task={task.task_id} type={task.task_type} version={task.version} "
            f"workflow={settings.workflow} model={settings.llm.model}"
        )
        try:
            visible_dir = settings.data_store.stage_inputs(task, staging_root)
            prediction_path = task_output / PREDICTION_FILENAME
            definition = build_task_definition(
                task,
                agent_visible_dir=visible_dir,
                output_path=prediction_path,
            )
            result, cost, usage = await evaluate_task(definition)
            record = _last_task_record(runner.get_run_records(), task.task_id)
            grade = score_prediction(task, prediction_path, settings.data_store)
            status = "completed" if grade.get("prediction_exist") else "failed"
            if grade.get("error") and grade.get("error") != "missing_target_columns":
                status = "failed"
            payload = {
                "schema_version": 1,
                "task_id": task.task_id,
                "base_task_id": task.base_task_id,
                "task_type": task.task_type,
                "version": task.version,
                "metric": task.metric,
                "status": status,
                "score": float(grade["final_score"]),
                "grade": grade,
                "prediction_path": str(prediction_path),
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
                "base_task_id": task.base_task_id,
                "task_type": task.task_type,
                "version": task.version,
                "metric": task.metric,
                "status": status,
                "score": 0.0,
                "grade": {"prediction_exist": 0.0, "final_score": 0.0, "error": str(exc)},
                "prediction_path": str(task_output / PREDICTION_FILENAME),
                "dslighting": {
                    "workflow": settings.workflow,
                    "model": settings.llm.model,
                    "cost": 0.0,
                    "usage": {},
                    "workspace_dir": "",
                },
            }
        (task_output / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{status} task={task.task_id} score={payload['score']:.6f} "
            f"error={payload['grade'].get('error')}"
        )
        return {
            "task_id": task.task_id,
            "status": status,
            "score": float(payload["score"]),
            "cost": float(payload["dslighting"]["cost"]),
            "result": str(task_output / "result.json"),
        }

    semaphore = asyncio.Semaphore(settings.concurrency)

    async def run_indexed(index: int, task: DareBenchTask):
        async with semaphore:
            return index, await run_one(task)

    try:
        indexed = await asyncio.gather(
            *(run_indexed(index, task) for index, task in enumerate(tasks))
        )
    finally:
        lock_handle.close()
    results = [result for _, result in sorted(indexed)]
    scores = [float(item.get("score") or 0.0) for item in results]
    summary = {
        "schema_version": 1,
        "benchmark": "dare-bench-fixed-gt-train",
        "workflow": settings.workflow,
        "model": config.llm.model,
        "thinking": config.llm.thinking,
        "sandbox": {
            "backend": settings.sandbox_backend,
            "local_isolation": settings.local_isolation
            if settings.sandbox_backend == "local"
            else None,
            "python_executable": str(config.sandbox.python_executable or ""),
            "network_disabled": settings.disable_network,
        },
        "task_manifest": str(selected_manifest),
        "task_count": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "average_reward": float(np.mean(scores)) if scores else 0.0,
        "total_cost": sum(float(item.get("cost") or 0.0) for item in results),
        "tasks": results,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "DareBenchTask",
    "DareDataStore",
    "PREDICTION_FILENAME",
    "RunSettings",
    "build_task_definition",
    "load_tasks",
    "run_tasks",
    "score_prediction",
    "select_tasks",
]
