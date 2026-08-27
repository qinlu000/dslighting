"""Run deterministic Data Agent RL tasks through DSLighting workflows."""

from __future__ import annotations

import ast
import asyncio
import fcntl
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import tomllib

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

ANSWER_FILENAME = "answer.txt"
SELECTED_TASKS_FILENAME = "data_agent_rl_tasks.jsonl"
DEFAULT_BUCKET_ID = "AdithyaSK/jupyter-agent-kaggle-all"
DETERMINISTIC_REWARD_MODES = frozenset({"exact_short", "numeric", "exact_bool", "list", "list_csv"})
MANIFEST_COLUMNS = {
    "task_dir",
    "task_id",
    "question",
    "answer",
    "reward_mode_initial",
    "difficulty_level",
    "package_tier",
    "kaggle_dataset_name",
    "files_used",
    "packages_used",
}
_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_TRUE_VALUES = {"yes", "true"}
_FALSE_VALUES = {"no", "false"}


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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


@dataclass(frozen=True)
class DataAgentTask:
    """One validated task from the Data Agent RL training manifest."""

    task_id: str
    source_task_id: str
    question: str
    gold_answer: str = field(repr=False)
    reward_mode: str
    difficulty_level: int
    package_tier: int
    kaggle_dataset_name: str
    bucket_prefix: str
    files_used: tuple[str, ...] = ()
    packages_used: tuple[str, ...] = ()

    @property
    def output_key(self) -> str:
        return self.task_id

    def manifest_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_task_id": self.source_task_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "reward_mode": self.reward_mode,
            "difficulty_level": self.difficulty_level,
            "package_tier": self.package_tier,
            "kaggle_dataset_name": self.kaggle_dataset_name,
            "bucket_prefix": self.bucket_prefix,
            "files_used": list(self.files_used),
            "packages_used": list(self.packages_used),
        }


def _task_from_row(row: Mapping[str, Any], tasks_root: Path) -> DataAgentTask:
    reward_mode = str(row["reward_mode_initial"]).strip()
    if reward_mode not in DETERMINISTIC_REWARD_MODES:
        raise ValueError(f"Task uses non-deterministic reward mode: {reward_mode!r}")
    task_id = _safe_component(str(row["task_dir"]).strip(), field_name="task_dir")
    task_toml = tasks_root / task_id / "task.toml"
    if not task_toml.is_file():
        raise ValueError(f"Missing task spec: {task_toml}")
    with task_toml.open("rb") as handle:
        spec = tomllib.load(handle)
    metadata = spec.get("metadata") or {}
    env = (spec.get("environment") or {}).get("env") or {}

    source_task_id = str(row["task_id"]).strip()
    question = str(row["question"]).strip()
    gold_answer = str(row["answer"]).strip()
    kaggle_name = str(row["kaggle_dataset_name"]).strip()
    bucket_prefix = _safe_component(
        str(env.get("BUCKET_PREFIX") or "").strip(),
        field_name=f"{task_id}.BUCKET_PREFIX",
    )
    if str(metadata.get("source_row_id") or "").strip() != source_task_id:
        raise ValueError(f"Manifest/TOML source id mismatch for {task_id}")
    if _normalize(str(metadata.get("gold_answer") or "")) != _normalize(gold_answer):
        raise ValueError(f"Manifest/TOML gold mismatch for {task_id}")
    if str(metadata.get("reward_mode_initial") or "").strip() != reward_mode:
        raise ValueError(f"Manifest/TOML reward mode mismatch for {task_id}")
    if str(metadata.get("kaggle_dataset_name") or "").strip() != kaggle_name:
        raise ValueError(f"Manifest/TOML dataset mismatch for {task_id}")
    if not question or not gold_answer:
        raise ValueError(f"Task {task_id} has an empty question or gold answer")
    return DataAgentTask(
        task_id=task_id,
        source_task_id=source_task_id,
        question=question,
        gold_answer=gold_answer,
        reward_mode=reward_mode,
        difficulty_level=int(row["difficulty_level"]),
        package_tier=int(row["package_tier"]),
        kaggle_dataset_name=kaggle_name,
        bucket_prefix=bucket_prefix,
        files_used=_string_tuple(row["files_used"]),
        packages_used=_string_tuple(row["packages_used"]),
    )


def load_tasks(
    dataset_root: Path,
    *,
    reward_modes: Iterable[str] = DETERMINISTIC_REWARD_MODES,
) -> list[DataAgentTask]:
    """Load the local manifest and validate it against each task TOML."""

    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "manifest.parquet"
    tasks_root = root / "tasks"
    if not manifest_path.is_file() or not tasks_root.is_dir():
        raise ValueError(f"Invalid Data Agent RL dataset root: {root}")

    allowed_modes = {str(mode) for mode in reward_modes}
    unsupported = allowed_modes - DETERMINISTIC_REWARD_MODES
    if unsupported:
        raise ValueError(f"Non-deterministic or unsupported reward modes: {sorted(unsupported)}")
    if not allowed_modes:
        raise ValueError("At least one deterministic reward mode is required")

    frame = pd.read_parquet(manifest_path)
    missing = MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Data Agent RL manifest is missing columns: {sorted(missing)}")

    tasks: list[DataAgentTask] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        reward_mode = str(row["reward_mode_initial"]).strip()
        if reward_mode not in allowed_modes:
            continue
        task = _task_from_row(row, tasks_root)
        task_id = task.task_id
        if task_id in seen:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError("No Data Agent RL tasks matched the deterministic reward modes")
    return tasks


def load_task(dataset_root: Path, task_id: str) -> DataAgentTask:
    """Load one task without parsing every task TOML in the dataset."""

    root = Path(dataset_root).expanduser().resolve()
    task_id = _safe_component(str(task_id).strip(), field_name="task_id")
    manifest_path = root / "manifest.parquet"
    tasks_root = root / "tasks"
    if not manifest_path.is_file() or not tasks_root.is_dir():
        raise ValueError(f"Invalid Data Agent RL dataset root: {root}")
    frame = pd.read_parquet(manifest_path, filters=[("task_dir", "==", task_id)])
    missing = MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Data Agent RL manifest is missing columns: {sorted(missing)}")
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one Data Agent task {task_id!r}, found {len(frame)}")
    return _task_from_row(frame.to_dict(orient="records")[0], tasks_root)


def select_tasks(
    tasks: Sequence[DataAgentTask],
    *,
    task_ids: Sequence[str] = (),
    index_expression: str | None = None,
    select_all: bool = False,
) -> list[DataAgentTask]:
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
class BucketDataStore:
    """Cache public HF Bucket datasets outside the network-disabled sandbox."""

    cache_root: Path
    bucket_id: str = DEFAULT_BUCKET_ID
    token: str | bool | None = None
    offline: bool = False

    def __post_init__(self) -> None:
        root = self.cache_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / ".locks").mkdir(exist_ok=True)
        (root / ".complete").mkdir(exist_ok=True)
        object.__setattr__(self, "cache_root", root)

    def _paths(self, task: DataAgentTask) -> tuple[Path, Path, Path]:
        data_dir = self.cache_root / task.bucket_prefix
        marker = self.cache_root / ".complete" / f"{task.bucket_prefix}.json"
        lock = self.cache_root / ".locks" / f"{task.bucket_prefix}.lock"
        return data_dir, marker, lock

    def cached_dataset(self, task: DataAgentTask) -> Path | None:
        """Return the complete local dataset path without performing network I/O."""

        data_dir, marker, _ = self._paths(task)
        required = {Path(value).name for value in task.files_used}
        return data_dir if self._cached_files(data_dir, marker, required) else None

    @staticmethod
    def _cached_files(
        data_dir: Path,
        marker: Path,
        required_names: set[str] | None = None,
    ) -> list[Path]:
        if not data_dir.is_dir() or not marker.is_file():
            return []
        try:
            names = json.loads(marker.read_text(encoding="utf-8"))["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return []
        if required_names and not required_names.issubset(set(names)):
            return []
        files = [data_dir / str(name) for name in names]
        return (
            files
            if files and all(path.is_file() and path.stat().st_size > 0 for path in files)
            else []
        )

    def ensure_dataset(self, task: DataAgentTask) -> Path:
        data_dir, marker, lock_path = self._paths(task)
        if self.cached_dataset(task) is not None:
            return data_dir

        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.cached_dataset(task) is not None:
                return data_dir
            if self.offline:
                raise ValueError(f"Dataset is not cached for offline use: {task.bucket_prefix}")

            from huggingface_hub import download_bucket_files, list_bucket_tree

            prefix = f"{task.bucket_prefix}/"
            remote = [
                item.path
                for item in list_bucket_tree(
                    self.bucket_id,
                    prefix=prefix,
                    recursive=True,
                    token=self.token,
                )
                if getattr(item, "type", None) == "file"
            ]
            if not remote:
                raise ValueError(f"No files found in hf://buckets/{self.bucket_id}/{prefix}")
            remote_by_name: dict[str, str] = {}
            for remote_path in remote:
                name = Path(remote_path).name
                current = remote_by_name.get(name)
                if current is None or (remote_path.count("/"), remote_path) < (
                    current.count("/"),
                    current,
                ):
                    remote_by_name[name] = remote_path
            required_names = {Path(value).name for value in task.files_used}
            missing_remote = sorted(required_names - remote_by_name.keys())
            if missing_remote:
                raise ValueError(
                    f"Required task files are missing from bucket prefix {prefix}: {missing_remote}"
                )
            selected_remote = [(remote_by_name[name], name) for name in sorted(required_names)]
            data_dir.mkdir(parents=True, exist_ok=True)
            targets = []
            for remote_path, name in selected_remote:
                _safe_component(name, field_name=f"{task.bucket_prefix}.bucket_file")
                destination = data_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or destination.stat().st_size == 0:
                    targets.append((remote_path, destination))
            if targets:
                download_bucket_files(
                    self.bucket_id,
                    files=targets,
                    raise_on_missing_files=True,
                    token=self.token,
                )
            if not all(path.is_file() and path.stat().st_size > 0 for _, path in targets):
                raise ValueError(f"Incomplete bucket download for {task.bucket_prefix}")
            existing_names: set[str] = set()
            if marker.is_file():
                try:
                    existing_names = set(json.loads(marker.read_text(encoding="utf-8"))["files"])
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            cached_names = sorted(existing_names | required_names)
            marker.write_text(
                json.dumps(
                    {
                        "bucket_id": self.bucket_id,
                        "prefix": task.bucket_prefix,
                        "files": cached_names,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.cached_dataset(task) is None:
                raise ValueError(f"Incomplete bucket cache for {task.bucket_prefix}")
            return data_dir

    def stage_inputs(self, task: DataAgentTask, staging_root: Path) -> Path:
        data_dir = self.ensure_dataset(task)
        stage_dir = Path(staging_root).expanduser().resolve() / task.output_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_link = stage_dir / "input"
        if input_link.is_symlink():
            if input_link.resolve() != data_dir:
                raise ValueError(f"Staging input points to the wrong dataset: {input_link}")
        elif input_link.exists():
            raise ValueError(f"Staging input path is not a symlink: {input_link}")
        else:
            os.symlink(data_dir, input_link, target_is_directory=True)
        return stage_dir


def build_task_definition(
    task: DataAgentTask,
    *,
    agent_visible_dir: Path,
    output_path: Path,
) -> TaskDefinition:
    output_path = Path(output_path).expanduser().resolve()
    io_instructions = (
        "All dataset files are in the read-only `input/` directory in the current working "
        "directory. Inspect the files and compute the answer with Python. Each Action may run "
        "in a fresh process, so persist anything important in files. Write only the short answer "
        f"value to `{ANSWER_FILENAME}` in the current working directory. Do not include labels, "
        "explanations, Markdown, or units unless the question requires them. Before finishing, "
        f"verify that `{ANSWER_FILENAME}` exists."
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
        "source_id": "data-agent-rl-environment",
        "engine_id": "data-agent-rl",
        "submission_artifact_contract": {
            "output_submission_path": str(output_path),
            "submission_filename": ANSWER_FILENAME,
            "submission_format": ".txt",
            "root_kind": "file",
            "root_basename": "answer",
            "entries": [
                {
                    "relative_path": ANSWER_FILENAME,
                    "format": "txt",
                    "description": "Short deterministic answer",
                }
            ],
        },
    }
    return TaskDefinition(
        task_id=task.task_id,
        task_type="datasci",
        payload={"execution_spec": execution_spec},
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _to_float(value: str) -> float | None:
    match = _NUMERIC_RE.search(str(value or "").replace(",", ""))
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _to_bool(value: str) -> bool | None:
    normalized = _normalize(value)
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _literal_sequence(value: str) -> tuple[Any, ...] | None:
    try:
        parsed = ast.literal_eval(str(value).strip())
    except (SyntaxError, ValueError):
        return None
    return tuple(parsed) if isinstance(parsed, (list, tuple)) else None


def _sequence_equal(expected: tuple[Any, ...], candidate: tuple[Any, ...]) -> bool:
    if len(expected) != len(candidate):
        return False
    for left, right in zip(expected, candidate):
        left_number, right_number = _to_float(str(left)), _to_float(str(right))
        if left_number is not None and right_number is not None:
            if not math.isclose(left_number, right_number, rel_tol=1e-3, abs_tol=1e-3):
                return False
        elif _normalize(str(left)) != _normalize(str(right)):
            return False
    return True


def grade_answer(task: DataAgentTask, answer_path: Path) -> dict[str, Any]:
    answer_path = Path(answer_path).expanduser().resolve()
    if not answer_path.is_file():
        return {"answer_exist": 0.0, "reward": 0.0, "method": "missing_answer"}
    try:
        candidate = answer_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {
            "answer_exist": 1.0,
            "reward": 0.0,
            "method": "unreadable_answer",
            "detail": str(exc),
        }
    if not candidate:
        return {"answer_exist": 0.0, "reward": 0.0, "method": "empty_answer"}
    if _normalize(task.gold_answer) == _normalize(candidate):
        return {"answer_exist": 1.0, "reward": 1.0, "method": "exact", "answer": candidate}

    matched = False
    method = "miss"
    if task.reward_mode == "numeric":
        gold, predicted = _to_float(task.gold_answer), _to_float(candidate)
        matched = (
            gold is not None
            and predicted is not None
            and math.isclose(gold, predicted, rel_tol=1e-3, abs_tol=1e-3)
        )
        method = "numeric" if matched else "miss"
    elif task.reward_mode == "exact_bool":
        gold, predicted = _to_bool(task.gold_answer), _to_bool(candidate)
        matched = gold is not None and predicted is not None and gold == predicted
        method = "bool" if matched else "miss"
    elif task.reward_mode in {"list", "list_csv"}:
        gold = _literal_sequence(task.gold_answer)
        predicted = _literal_sequence(candidate)
        matched = gold is not None and predicted is not None and _sequence_equal(gold, predicted)
        method = "list" if matched else "miss"
    return {
        "answer_exist": 1.0,
        "reward": 1.0 if matched else 0.0,
        "method": method,
        "answer": candidate,
    }


@dataclass(frozen=True)
class RunSettings:
    data_store: BucketDataStore
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
    download_concurrency: int = 8
    timeout_seconds: int = 600
    local_isolation: str = "bubblewrap"
    sandbox_python: Path | None = None
    compute_threads: int = 2
    overwrite: bool = False
    retry_failed: bool = False


async def run_tasks(tasks: Sequence[DataAgentTask], settings: RunSettings) -> dict[str, Any]:
    if not tasks:
        raise ValueError("At least one Data Agent RL task must be selected")
    if (
        min(
            settings.concurrency,
            settings.download_concurrency,
            settings.max_steps,
            settings.timeout_seconds,
            settings.compute_threads,
        )
        <= 0
    ):
        raise ValueError("Concurrency, steps, timeout, and compute threads must be positive")
    if settings.summary_trigger_turns < settings.keep_recent_turns:
        raise ValueError("summary_trigger_turns must be >= keep_recent_turns")

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
            tensorflow=True,
        )
    )
    config = runner.config
    evaluate_task = runner.get_eval_function()
    download_semaphore = asyncio.Semaphore(settings.download_concurrency)

    async def run_one(task: DataAgentTask) -> dict[str, Any]:
        task_output = output_root / task.output_key
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

        print(
            f"run task={task.task_id} difficulty=L{task.difficulty_level} "
            f"mode={task.reward_mode} workflow={settings.workflow} model={settings.llm.model}"
        )
        answer_path = task_output / ANSWER_FILENAME
        try:
            async with download_semaphore:
                visible_dir = await asyncio.to_thread(
                    settings.data_store.stage_inputs,
                    task,
                    staging_root,
                )
            definition = build_task_definition(
                task,
                agent_visible_dir=visible_dir,
                output_path=answer_path,
            )
            result, cost, usage = await evaluate_task(definition)
            if isinstance(result, str) and result.startswith("[ERROR]"):
                raise RuntimeError(result)
            record = last_task_record(runner.get_run_records(), task.task_id)
            grade = grade_answer(task, answer_path)
            status = "completed" if grade.get("answer_exist") else "failed"
            payload = {
                "schema_version": 1,
                "task_id": task.task_id,
                "source_task_id": task.source_task_id,
                "difficulty_level": task.difficulty_level,
                "package_tier": task.package_tier,
                "reward_mode": task.reward_mode,
                "status": status,
                "reward": float(grade["reward"]),
                "grade": grade,
                "answer_path": str(answer_path),
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
                "source_task_id": task.source_task_id,
                "difficulty_level": task.difficulty_level,
                "package_tier": task.package_tier,
                "reward_mode": task.reward_mode,
                "status": status,
                "reward": 0.0,
                "grade": {
                    "answer_exist": 0.0,
                    "reward": 0.0,
                    "method": "error",
                    "detail": str(exc),
                },
                "answer_path": str(answer_path),
                "dslighting": {
                    "workflow": settings.workflow,
                    "model": settings.llm.model,
                    "cost": 0.0,
                    "usage": {},
                    "workspace_dir": "",
                },
            }
        write_json(task_output / "result.json", payload)
        print(
            f"{status} task={task.task_id} reward={payload['reward']:.1f} "
            f"method={payload['grade'].get('method')}"
        )
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
        "benchmark": "data-agent-rl-environment-train-deterministic",
        "workflow": settings.workflow,
        "model": config.llm.model,
        "thinking": config.llm.thinking,
        "sandbox": {
            "backend": "local",
            "local_isolation": settings.local_isolation,
            "python_executable": str(config.sandbox.python_executable or ""),
            "network_disabled": True,
            "compute_threads": settings.compute_threads,
        },
        "bucket_cache": str(settings.data_store.cache_root),
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
    "ANSWER_FILENAME",
    "DETERMINISTIC_REWARD_MODES",
    "BucketDataStore",
    "DataAgentTask",
    "RunSettings",
    "build_task_definition",
    "grade_answer",
    "load_task",
    "load_tasks",
    "run_tasks",
    "select_tasks",
]
