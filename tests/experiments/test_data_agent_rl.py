from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.data_agent_rl.runner import (
    BucketDataStore,
    DataAgentTask,
    build_task_definition,
    grade_answer,
    load_task,
    load_tasks,
    select_tasks,
)


def _task(**overrides) -> DataAgentTask:
    values = {
        "task_id": "0001_001_1001_qa_1",
        "source_task_id": "0001/001/1001.ipynb_qa_1",
        "question": "How many rows are present?",
        "gold_answer": "11",
        "reward_mode": "numeric",
        "difficulty_level": 1,
        "package_tier": 1,
        "kaggle_dataset_name": "owner/dataset",
        "bucket_prefix": "owner__dataset",
    }
    values.update(overrides)
    return DataAgentTask(**values)


def test_load_tasks_filters_non_deterministic_modes(tmp_path: Path) -> None:
    task_id = "0001_001_1001_qa_1"
    task_dir = tmp_path / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        """
[metadata]
source_row_id = "0001/001/1001.ipynb_qa_1"
kaggle_dataset_name = "owner/dataset"
gold_answer = "11"
reward_mode_initial = "numeric"

[environment.env]
BUCKET_PREFIX = "owner__dataset"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "task_dir": task_id,
            "task_id": "0001/001/1001.ipynb_qa_1",
            "question": "How many rows are present?",
            "answer": "11",
            "reward_mode_initial": "numeric",
            "difficulty_level": 1,
            "package_tier": 1,
            "kaggle_dataset_name": "owner/dataset",
            "files_used": ["/kaggle/input/dataset/data.csv"],
            "packages_used": ["pandas"],
        },
        {
            "task_dir": "not_downloaded_flexible_task",
            "task_id": "other",
            "question": "Flexible?",
            "answer": "value and context",
            "reward_mode_initial": "flexible",
            "difficulty_level": 1,
            "package_tier": 1,
            "kaggle_dataset_name": "owner/other",
            "files_used": [],
            "packages_used": [],
        },
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "manifest.parquet")

    tasks = load_tasks(tmp_path)

    assert [task.task_id for task in tasks] == [task_id]
    assert tasks[0].files_used == ("/kaggle/input/dataset/data.csv",)
    assert load_task(tmp_path, task_id) == tasks[0]


def test_select_tasks_uses_filtered_order() -> None:
    tasks = [_task(task_id=f"task_{index}") for index in range(4)]

    assert [task.task_id for task in select_tasks(tasks, index_expression="1-3")] == [
        "task_1",
        "task_2",
    ]
    with pytest.raises(ValueError, match="Choose exactly one"):
        select_tasks(tasks, index_expression="0-1", select_all=True)


@pytest.mark.parametrize(
    ("task", "candidate", "method"),
    [
        (_task(reward_mode="exact_short", gold_answer="Mumbai Indians"), "mumbai  indians", "exact"),
        (_task(gold_answer="0.3918"), "The result is 0.39181", "numeric"),
        (_task(reward_mode="exact_bool", gold_answer="Yes"), "true", "bool"),
        (
            _task(reward_mode="list", gold_answer="[35.69927362, 139.68024829]"),
            "(35.6992736, 139.6802483)",
            "list",
        ),
    ],
)
def test_grade_answer_is_deterministic(
    tmp_path: Path,
    task: DataAgentTask,
    candidate: str,
    method: str,
) -> None:
    answer = tmp_path / "answer.txt"
    answer.write_text(candidate, encoding="utf-8")

    grade = grade_answer(task, answer)

    assert grade["reward"] == 1.0
    assert grade["method"] == method


def test_grade_answer_reports_missing_artifact(tmp_path: Path) -> None:
    assert grade_answer(_task(), tmp_path / "answer.txt") == {
        "answer_exist": 0.0,
        "reward": 0.0,
        "method": "missing_answer",
    }


def test_bucket_cache_stages_one_read_only_input_link(tmp_path: Path) -> None:
    task = _task()
    cache = tmp_path / "cache"
    data_dir = cache / task.bucket_prefix
    data_dir.mkdir(parents=True)
    (data_dir / "data.csv").write_text("value\n11\n", encoding="utf-8")
    complete = cache / ".complete"
    complete.mkdir()
    (complete / f"{task.bucket_prefix}.json").write_text(
        json.dumps({"files": ["data.csv"]}),
        encoding="utf-8",
    )
    store = BucketDataStore(cache_root=cache, offline=True)

    stage = store.stage_inputs(task, tmp_path / "staging")

    assert (stage / "input").is_symlink()
    assert (stage / "input" / "data.csv").read_text(encoding="utf-8") == "value\n11\n"


def test_bucket_cache_requires_every_file_used_by_each_task(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    data_dir = cache / "owner__dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "first.csv").write_text("value\n1\n", encoding="utf-8")
    marker = cache / ".complete"
    marker.mkdir()
    (marker / "owner__dataset.json").write_text(
        json.dumps({"files": ["first.csv"]}), encoding="utf-8"
    )
    store = BucketDataStore(cache_root=cache, offline=True)
    first = _task(files_used=("/kaggle/input/dataset/first.csv",))
    second = _task(files_used=("/kaggle/input/dataset/second.csv",))

    assert store.cached_dataset(first) == data_dir
    assert store.cached_dataset(second) is None


def test_task_definition_never_contains_gold(tmp_path: Path) -> None:
    task = _task(gold_answer="SECRET_GOLD")

    definition = build_task_definition(
        task,
        agent_visible_dir=tmp_path,
        output_path=tmp_path / "answer.txt",
    )

    serialized = json.dumps(definition.payload)
    assert "SECRET_GOLD" not in serialized
    assert "answer.txt" in serialized
    assert "input/" in serialized
