"""Fail-fast gate for the two deterministic OOD benchmarks on HPC."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from dslighting.benchmark.vendor.dabench.registry import Registry
from dslighting.benchmark.vendor.mlebench.data import is_dataset_prepared
from experiments.agenticdatabench.runner import load_tasks, resolve_domain_dir
from experiments.training_mix.hpc.preflight_compute import _bubblewrap_smoke


def _check_dabench(data_root: Path) -> dict[str, Any]:
    registry = Registry(data_root)
    task_ids = registry.list_competition_ids()
    failures: list[str] = []
    for task_id in task_ids:
        try:
            competition = registry.get_competition(task_id)
            if not is_dataset_prepared(competition):
                raise ValueError("prepared data contract failed")
            answers = pd.read_csv(competition.answers)
            if competition.grader(answers.copy(), answers) != 1.0:
                raise ValueError("gold self-score is not 1.0")
        except Exception as exc:
            failures.append(f"{task_id}: {exc}")
    if len(task_ids) != 257:
        failures.append(f"expected 257 public tasks, found {len(task_ids)}")
    return {"tasks": len(task_ids), "failures": failures, "passed": not failures}


def _check_agentic(benchmark_root: Path, python: Path) -> dict[str, Any]:
    testbed = benchmark_root / "testbed"
    tasks = load_tasks(testbed / "tasks" / "dev.jsonl")
    dataset_root = testbed / "datasets"
    gold_root = testbed / "gold"
    failures: list[str] = []
    for task in tasks:
        try:
            resolve_domain_dir(dataset_root, task.domain)
            names = task.upstream_record.get("gold_file_name")
            names = names if isinstance(names, list) else [names]
            for name in names:
                path = gold_root / task.task_id / Path(str(name)).name
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
        except Exception as exc:
            failures.append(f"{task.task_id}: {exc}")
    if len(tasks) != 246:
        failures.append(f"expected 246 public tasks, found {len(tasks)}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(testbed)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import cv2, duckdb, geopandas, lightgbm, pandas, sklearn, xgboost; "
                "from da_agent.evaluators.evaluation import Evaluator"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    if completed.returncode:
        failures.append(f"official evaluator import failed: {completed.stderr.strip()}")
    return {"tasks": len(tasks), "failures": failures, "passed": not failures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-python", type=Path, required=True)
    parser.add_argument("--dabench-python", type=Path, required=True)
    parser.add_argument("--agentic-python", type=Path, required=True)
    parser.add_argument("--dabench-root", type=Path, required=True)
    parser.add_argument("--agentic-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = {
            "dabench": _check_dabench(args.dabench_root.expanduser().resolve()),
            "agenticdatabench": _check_agentic(
                args.agentic_root.expanduser().resolve(),
                args.agentic_python.expanduser().absolute(),
            ),
            "bubblewrap": {
                "dabench": _bubblewrap_smoke(args.controller_python, args.dabench_python),
                "agenticdatabench": _bubblewrap_smoke(
                    args.controller_python, args.agentic_python
                ),
            },
        }
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    passed = all(
        (
            report["dabench"]["passed"],
            report["agenticdatabench"]["passed"],
            report["bubblewrap"]["dabench"]["passed"],
            report["bubblewrap"]["agenticdatabench"]["passed"],
        )
    )
    report["status"] = "passed" if passed else "failed"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
