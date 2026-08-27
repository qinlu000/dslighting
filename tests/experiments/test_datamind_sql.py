from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from experiments.datamind_sql.runner import (
    DataMindSQLDataStore,
    build_task_definition,
    grade_result,
    load_tasks,
    tables_match,
)


def _record() -> dict:
    return {
        "source": "datamind_sql",
        "task_id": "example_0",
        "question": "Return each value.",
        "task_family": "sql",
        "dataset_key": "example",
        "input_ref": {"kind": "sqlite", "ref": "rl/train_files/example.sqlite"},
        "evaluator": {
            "kind": "csv_table_match",
            "ignore_order": True,
            "ground_truth_ref": "rl/gold_csv_results/example_0.csv",
            "agent_visible": False,
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, DataMindSQLDataStore]:
    manifest = tmp_path / "mix.jsonl"
    manifest.write_text(
        json.dumps({"source": "dare_bench", "task_id": "other"})
        + "\n"
        + json.dumps(_record())
        + "\n",
        encoding="utf-8",
    )
    database = tmp_path / "data" / "rl" / "train_files" / "example.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("create table values_table (value real)")
        connection.executemany("insert into values_table values (?)", [(1.0,), (2.0,)])
    gold = tmp_path / "data" / "rl" / "gold_csv_results" / "example_0.csv"
    gold.parent.mkdir(parents=True)
    pd.DataFrame({"value": [1.0, 2.0]}).to_csv(gold, index=False)
    return manifest, DataMindSQLDataStore(tmp_path / "data")


def test_mixed_manifest_staging_and_contract_hide_gold(tmp_path: Path) -> None:
    manifest, store = _fixture(tmp_path)
    task = load_tasks(manifest)[0]
    visible = store.stage_inputs(task, tmp_path / "staging")
    definition = build_task_definition(
        task, agent_visible_dir=visible, output_path=tmp_path / "result.csv"
    )

    assert (visible / "database.sqlite").is_symlink()
    assert not (visible / "example_0.csv").exists()
    serialized = json.dumps(definition.payload)
    assert "gold_csv_results" not in serialized
    assert "result.csv" in serialized


def test_grade_result_matches_official_columnwise_semantics(tmp_path: Path) -> None:
    manifest, store = _fixture(tmp_path)
    task = load_tasks(manifest)[0]
    prediction = tmp_path / "result.csv"
    pd.DataFrame({"renamed": [2.005, 1.0], "extra": [9, 9]}).to_csv(prediction, index=False)

    grade = grade_result(task, prediction, store)

    assert grade["reward"] == 1.0
    assert grade["method"] == "csv_table_match"


def test_table_match_rejects_wrong_values() -> None:
    gold = pd.DataFrame({"answer": [1, 2]})
    prediction = pd.DataFrame({"answer": [1, 3]})

    assert not tables_match(prediction, gold, ignore_order=True)
