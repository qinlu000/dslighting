from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.training_mix.hpc.ood_smoke import _summarize_dabench_results


def _write_scores(path: Path, scores: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["competition_id", "score"])
        writer.writeheader()
        for index, score in enumerate(scores):
            writer.writerow({"competition_id": f"task-{index}", "score": score})


def test_summarize_dabench_results_requires_complete_numeric_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.csv"
    _write_scores(path, [1.0, 0.5, 0.0])

    assert _summarize_dabench_results(path, 3) == {
        "tasks": 3,
        "average_score": 0.5,
        "perfect_scores": 1,
        "zero_scores": 1,
    }

    with pytest.raises(RuntimeError, match="2 selected tasks"):
        _summarize_dabench_results(path, 2)
