"""Freeze a deterministic explore/test split of public AgenticDataBench tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EXPLORE_FRACTION = 0.25
DEFAULT_SPLIT_SALT = "datacope-agenticdatabench-v1"
DEFAULT_SPLIT_MANIFEST = Path(__file__).with_name("split.json")
DEFAULT_TASKS_RELATIVE_PATH = Path("testbed/tasks/dev.jsonl")
DEFAULT_DATASETS_RELATIVE_PATH = Path("testbed/datasets")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    question: str
    domain: str
    output_file_names: tuple[str, ...]
    has_plot_post_process: bool


def _path_component(value: str, *, field: str) -> str:
    value = str(value or "").strip()
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be one safe path component: {value!r}")
    return value


def load_task_records(path: Path) -> tuple[TaskRecord, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"AgenticDataBench task file does not exist: {source}")

    records: list[TaskRecord] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")

            task_id = _path_component(payload.get("id"), field="id")
            if task_id in seen:
                raise ValueError(f"Duplicate AgenticDataBench task ID: {task_id}")
            question = str(payload.get("question") or "").strip()
            domain = str(payload.get("domain") or "").strip()
            raw_outputs = payload.get("output_file_name")
            outputs = [raw_outputs] if isinstance(raw_outputs, str) else raw_outputs
            if not question or not domain or not isinstance(outputs, list) or not outputs:
                raise ValueError(f"Incomplete AgenticDataBench task: {task_id}")
            output_names = tuple(
                _path_component(value, field=f"{task_id}.output_file_name")
                for value in outputs
            )
            if len(set(output_names)) != len(output_names):
                raise ValueError(f"Duplicate output file for task: {task_id}")
            post_process = payload.get("post_process_func") or []
            post_process = [post_process] if isinstance(post_process, str) else post_process
            records.append(
                TaskRecord(
                    task_id=task_id,
                    question=question,
                    domain=domain,
                    output_file_names=output_names,
                    has_plot_post_process=any(
                        str(value).strip().startswith("image_post_process(")
                        for value in post_process
                    ),
                )
            )
            seen.add(task_id)
    if len(records) < 2:
        raise ValueError("AgenticDataBench split requires at least two public tasks")
    return tuple(records)


def required_output_names(task: TaskRecord) -> tuple[str, ...]:
    required = list(task.output_file_names)
    if task.has_plot_post_process:
        for output_name in task.output_file_names:
            path = Path(output_name)
            if path.suffix.lower() != ".png":
                continue
            for suffix in (".json", ".npy"):
                sidecar = f"{path.stem}{suffix}"
                if sidecar not in required:
                    required.append(sidecar)
    return tuple(required)


def task_source_sha256(tasks_file: Path) -> str:
    return hashlib.sha256(Path(tasks_file).expanduser().resolve().read_bytes()).hexdigest()


@dataclass(frozen=True)
class AgenticDataBenchSplit:
    manifest_path: Path
    split_id: str
    source_sha256: str
    explore_task_ids: tuple[str, ...]
    test_task_ids: tuple[str, ...]

    def task_ids(self, phase: str) -> tuple[str, ...]:
        if phase == "explore":
            return self.explore_task_ids
        if phase == "test":
            return self.test_task_ids
        raise ValueError(f"Unknown phase: {phase}")

    def query_ids(self, phase: str) -> set[str]:
        return set(self.task_ids(phase))

    def verify_source(self, tasks_file: Path) -> None:
        tasks = load_task_records(tasks_file)
        if task_source_sha256(tasks_file) != self.source_sha256:
            raise ValueError(f"AgenticDataBench task source changed: {Path(tasks_file).resolve()}")
        if {task.task_id for task in tasks} != set((*self.explore_task_ids, *self.test_task_ids)):
            raise ValueError("AgenticDataBench task IDs differ from the frozen split")


def create_split(
    tasks_file: Path,
    output_path: Path = DEFAULT_SPLIT_MANIFEST,
    *,
    explore_fraction: float = DEFAULT_EXPLORE_FRACTION,
    salt: str = DEFAULT_SPLIT_SALT,
) -> Path:
    if not 0 < explore_fraction < 1:
        raise ValueError("explore_fraction must be between 0 and 1")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Split manifest already exists: {output}")

    tasks = load_task_records(tasks_file)
    ordered = sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{salt}:{task.task_id}".encode()).hexdigest(),
    )
    count = max(1, min(len(tasks) - 1, int(len(tasks) * explore_fraction + 0.5)))
    explore_ids = {task.task_id for task in ordered[:count]}
    payload = {
        "schema_version": 1,
        "benchmark": "AgenticDataBench",
        "split_id": salt,
        "explore_fraction": explore_fraction,
        "source_sha256": task_source_sha256(tasks_file),
        "explore_task_ids": [task.task_id for task in tasks if task.task_id in explore_ids],
        "test_task_ids": [task.task_id for task in tasks if task.task_id not in explore_ids],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_split(path: Path = DEFAULT_SPLIT_MANIFEST) -> AgenticDataBenchSplit:
    manifest = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload["schema_version"] != 1 or payload["benchmark"] != "AgenticDataBench":
            raise ValueError("unsupported manifest")
        explore = tuple(str(value) for value in payload["explore_task_ids"])
        test = tuple(str(value) for value in payload["test_task_ids"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid AgenticDataBench split manifest {manifest}: {exc}") from exc
    if not explore or not test or len(set(explore)) != len(explore) or len(set(test)) != len(test):
        raise ValueError("AgenticDataBench split partitions must be non-empty and unique")
    if set(explore) & set(test):
        raise ValueError("AgenticDataBench explore/test partitions overlap")
    return AgenticDataBenchSplit(
        manifest_path=manifest,
        split_id=str(payload["split_id"]),
        source_sha256=str(payload["source_sha256"]),
        explore_task_ids=explore,
        test_task_ids=test,
    )
