"""Materialize the upstream DABench release for the DSLighting registry."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from dslighting.benchmark.vendor.dabench.registry import Registry
from dslighting.benchmark.vendor.mlebench.data import is_dataset_prepared


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def materialize(source_root: Path, output_root: Path) -> int:
    questions = _read_jsonl(source_root / "da-dev-questions.jsonl")
    labels = {row["id"]: row for row in _read_jsonl(source_root / "da-dev-labels.jsonl")}
    tables = source_root / "da-dev-tables"

    registry = Registry(output_root)
    task_ids: dict[int, str] = {}
    for task_id in registry.list_competition_ids():
        match = re.fullmatch(r"dabench-(\d+)-.+", task_id)
        if match:
            task_ids[int(match.group(1))] = task_id

    question_ids = {row["id"] for row in questions}
    if question_ids != labels.keys() or question_ids != task_ids.keys():
        raise ValueError("Question, label, and DSLighting registry task IDs do not match")

    output_root.mkdir(parents=True, exist_ok=True)
    for index, question in enumerate(questions, start=1):
        question_id = question["id"]
        task_id = task_ids[question_id]
        competition = registry.get_competition(task_id)

        competition.raw_dir.mkdir(parents=True, exist_ok=True)
        competition.public_dir.mkdir(parents=True, exist_ok=True)
        competition.private_dir.mkdir(parents=True, exist_ok=True)

        source_table = tables / question["file_name"]
        if not source_table.is_file():
            raise FileNotFoundError(source_table)
        shutil.copy2(source_table, competition.raw_dir / source_table.name)

        with contextlib.redirect_stdout(io.StringIO()):
            competition.prepare_fn(
                raw=competition.raw_dir,
                public=competition.public_dir,
                private=competition.private_dir,
            )
        (competition.public_dir / "description.md").write_text(
            competition.description,
            encoding="utf-8",
        )

        expected = " ".join(
            f"@{name}[{value}]" for name, value in labels[question_id]["common_answers"]
        )
        expected_frame = pd.DataFrame({"id": [question_id], "answer": [expected]})
        answer_frame = pd.read_csv(competition.answers)
        if competition.grader(expected_frame, answer_frame) != 1.0:
            raise ValueError(f"Prepared answer disagrees with upstream label: {task_id}")
        if not is_dataset_prepared(competition):
            raise ValueError(f"Prepared task failed the DSLighting data contract: {task_id}")

        if index % 25 == 0 or index == len(questions):
            print(f"Materialized {index}/{len(questions)} tasks")

    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = (args.output_root or source_root).expanduser().resolve()
    count = materialize(source_root, output_root)
    print(f"DABench data ready at {output_root} ({count} tasks)")


if __name__ == "__main__":
    main()
