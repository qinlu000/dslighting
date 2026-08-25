from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from experiments.dare_bench.runner import (
    DareDataStore,
    build_task_definition,
    load_tasks,
    score_prediction,
    select_tasks,
)


def _record(*, task_type: str = "classification", version: str = "v2") -> dict:
    base = f"example_{task_type}"
    metric = "macro_f1" if task_type == "classification" else "clipped_r2"
    gt_name = "ground_truth_v1.csv" if version == "v1" else "ground_truth.csv"
    if task_type == "time_series_analysis" and version == "v2":
        gt_name = "ground_truth_v2.csv"
    return {
        "task_id": f"{base}::{version}",
        "base_task_id": base,
        "split": "train",
        "version": version,
        "task_type": task_type,
        "agent_payload": {
            "question": "Train a model and write prediction.csv.",
            "available_tools": ["python_executor"],
            "database_relpath": base,
            "needed_files": ["train.csv", "val.csv", "metadata.txt"],
            "needed_file_members": [
                f"{base}/source/train.csv",
                f"{base}/source/val.csv",
                f"{base}/source/metadata.txt",
            ],
        },
        "metadata": {
            "all_metadata_relpath": f"{base}/verify/all_metadata.json",
            "target": ["target"],
            "save_file_type": "csv",
        },
        "grader": {
            "type": "deterministic",
            "metric": metric,
            "ground_truth_relpath": f"{base}/verify/{gt_name}",
            "ground_truth_columns": ["row_id", "target"],
            "prediction_filename": "prediction.csv",
        },
    }


def _write_fixture(tmp_path: Path, record: dict) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    archive_path = tmp_path / "databases.zip"
    base = record["base_task_id"]
    gt_member = record["grader"]["ground_truth_relpath"]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{base}/source/train.csv", "row_id,x,target\n0,1,a\n1,2,b\n")
        archive.writestr(f"{base}/source/val.csv", "row_id,x\n0,1\n1,2\n")
        archive.writestr(f"{base}/source/metadata.txt", "target: target\n")
        archive.writestr(gt_member, "row_id,target\n0,a\n1,b\n")
    return manifest, archive_path


def test_manifest_stage_and_task_contract_hide_ground_truth(tmp_path: Path) -> None:
    record = _record()
    manifest, archive_path = _write_fixture(tmp_path, record)
    task = load_tasks(manifest)[0]
    store = DareDataStore(databases_zip=archive_path)
    visible = store.stage_inputs(task, tmp_path / "staging")

    assert sorted(path.name for path in visible.iterdir()) == [
        "metadata.txt",
        "train.csv",
        "val.csv",
    ]
    assert not any("ground_truth" in path.name for path in visible.iterdir())

    definition = build_task_definition(
        task,
        agent_visible_dir=visible,
        output_path=tmp_path / "output" / "prediction.csv",
    )
    serialized = json.dumps(definition.payload)
    assert "ground_truth" not in serialized
    assert definition.payload["execution_spec"]["output_path"].endswith("prediction.csv")


def test_official_equivalent_classification_score(tmp_path: Path) -> None:
    manifest, archive_path = _write_fixture(tmp_path, _record())
    task = load_tasks(manifest)[0]
    store = DareDataStore(databases_zip=archive_path)
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame({"row_id": [0, 1], "target": ["a", "b"]}).to_csv(prediction, index=False)

    grade = score_prediction(task, prediction, store)

    assert grade["prediction_exist"] == 1.0
    assert grade["final_score"] == pytest.approx(1.0)
    assert grade["id_columns"] == ["row_id"]


def test_score_rejects_incomplete_alignment(tmp_path: Path) -> None:
    manifest, archive_path = _write_fixture(tmp_path, _record())
    task = load_tasks(manifest)[0]
    store = DareDataStore(databases_zip=archive_path)
    prediction = tmp_path / "prediction.csv"
    pd.DataFrame({"row_id": [0], "target": ["a"]}).to_csv(prediction, index=False)

    grade = score_prediction(task, prediction, store)

    assert grade["final_score"] == 0.0
    assert grade["error"] == "row_count_mismatch_after_merge"


def test_filter_selection_keeps_versioned_task_ids(tmp_path: Path) -> None:
    records = [_record(), _record(task_type="time_series_analysis", version="v1")]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    tasks = load_tasks(manifest)

    selected = select_tasks(tasks, task_ids=[records[1]["task_id"]])

    assert [task.task_id for task in selected] == [records[1]["task_id"]]
