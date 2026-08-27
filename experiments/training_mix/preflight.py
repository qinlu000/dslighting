"""Fail-fast local preflight for the frozen research training mixture."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.common import sha256_file
from experiments.dare_bench import runner as dare
from experiments.data_agent_rl import runner as data_agent
from experiments.datamind_sql import runner as datamind

ALLOWED_SOURCES = {"dare_bench", "data_agent_rl", "datamind_sql"}
DARE_PACKAGES = {
    "catboost": "1.2.3",
    "lightgbm": "4.3.0",
    "matplotlib": "3.10.9",
    "numpy": "1.26.3",
    "openpyxl": "3.1.2",
    "pandas": "2.2.0",
    "pmdarima": "2.0.4",
    "plotly": "6.9.0",
    "pyarrow": "15.0.0",
    "scikit-learn": "1.4.0",
    "scipy": "1.11.4",
    "statsmodels": "0.14.1",
    "transformers": "4.44.2",
    "xgboost": "2.0.3",
}
GENERAL_PACKAGES = {
    "catboost": "1.2.10",
    "category-encoders": "2.10.0",
    "h5py": "3.14.0",
    "imbalanced-learn": "0.14.2",
    "lightgbm": "4.7.0",
    "matplotlib": "3.11.1",
    "missingno": "0.5.2",
    "nltk": "3.10.3",
    "numpy": "2.5.2",
    "openpyxl": "3.1.5",
    "pandas": "3.0.5",
    "plotly": "6.9.0",
    "pyarrow": "25.0.1",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.1",
    "seaborn": "0.13.2",
    "statsmodels": "0.14.6",
    "tabulate": "0.10.0",
    "tensorflow": "2.21.0",
    "wordcloud": "1.9.6",
    "xgboost": "3.4.1",
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Training manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        source, task_id = str(row.get("source") or ""), str(row.get("task_id") or "")
        if source not in ALLOWED_SOURCES or not task_id:
            raise ValueError(f"Invalid source/task_id on {path}:{line_number}")
        key = (source, task_id)
        if key in keys:
            raise ValueError(f"Duplicate source task key: {key}")
        evaluator = row.get("evaluator")
        if not isinstance(evaluator, Mapping) or evaluator.get("agent_visible") is not False:
            raise ValueError(f"Ground truth visibility is unsafe for {key}")
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError(f"Training manifest is empty: {path}")
    return rows


def _environment_fingerprint(
    executable: Path,
    *,
    expected_python: tuple[int, int],
    packages: Mapping[str, str],
) -> dict[str, Any]:
    executable = executable.expanduser().absolute()
    if not executable.is_file():
        return {"passed": False, "executable": str(executable), "error": "missing executable"}
    program = """
import importlib.metadata as metadata
import json
import platform
import sys
names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "python": platform.python_version(),
    "python_series": [sys.version_info.major, sys.version_info.minor],
    "platform": platform.platform(),
    "packages": versions,
}))
"""
    try:
        completed = subprocess.run(
            [str(executable), "-c", program, json.dumps(list(packages))],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
    except Exception as exc:
        return {"passed": False, "executable": str(executable), "error": str(exc)}
    missing = sorted(name for name, version in payload["packages"].items() if version is None)
    mismatched = {
        name: {"expected": expected, "actual": payload["packages"].get(name)}
        for name, expected in packages.items()
        if payload["packages"].get(name) is not None and payload["packages"].get(name) != expected
    }
    payload.update(
        {
            "passed": (
                payload["python_series"] == list(expected_python) and not missing and not mismatched
            ),
            "executable": str(executable),
            "expected_python_series": list(expected_python),
            "missing_packages": missing,
            "mismatched_packages": mismatched,
        }
    )
    return payload


def _repository_fingerprint(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = git("status", "--porcelain")
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "changed_paths": len(status.splitlines()) if status else 0,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest)
    source_counts = Counter(str(row["source"]) for row in rows)
    selected_ids = {
        source: {str(row["task_id"]) for row in rows if row["source"] == source}
        for source in ALLOWED_SOURCES
    }
    blockers: list[str] = []
    warnings: list[str] = []

    dare_store = dare.DareDataStore(databases_dir=args.dare_databases_dir)
    dare_by_id = {task.task_id: task for task in dare.load_tasks(args.dare_manifest)}
    missing_dare_tasks = sorted(selected_ids["dare_bench"] - dare_by_id.keys())
    dare_failures: list[dict[str, Any]] = []
    if not missing_dare_tasks:
        for task_id in sorted(selected_ids["dare_bench"]):
            task = dare_by_id[task_id]
            try:
                for member in task.needed_file_members:
                    source_path = (dare_store.databases_dir / member).resolve()
                    source_path.relative_to(dare_store.databases_dir)
                    if not source_path.is_file() or source_path.stat().st_size == 0:
                        raise ValueError(f"Missing or empty DARE input: {source_path}")
                ground_truth_path = dare_store.databases_dir / task.ground_truth_relpath
                grade = dare.score_prediction(task, ground_truth_path, dare_store)
                if grade.get("final_score") != 1.0:
                    dare_failures.append({"task_id": task_id, "grade": grade})
            except Exception as exc:
                dare_failures.append({"task_id": task_id, "error": str(exc)})
    if missing_dare_tasks or dare_failures:
        blockers.append("DARE task artifacts or ground-truth self-scores are incomplete")

    agent_tasks = data_agent.load_tasks(args.data_agent_root)
    agent_by_id = {task.task_id: task for task in agent_tasks}
    missing_agent_tasks = sorted(selected_ids["data_agent_rl"] - agent_by_id.keys())
    cache = data_agent.BucketDataStore(cache_root=args.data_agent_cache, offline=True)
    missing_cache_prefixes = sorted(
        {
            agent_by_id[task_id].bucket_prefix
            for task_id in selected_ids["data_agent_rl"] - set(missing_agent_tasks)
            if cache.cached_dataset(agent_by_id[task_id]) is None
        }
    )
    if missing_agent_tasks:
        blockers.append("Selected Data Agent task specs are missing")
    if missing_cache_prefixes:
        message = f"Data Agent cache is missing {len(missing_cache_prefixes)} dataset prefixes"
        (warnings if args.allow_missing_cache else blockers).append(message)

    datamind_store = datamind.DataMindSQLDataStore(args.datamind_root)
    datamind_tasks = datamind.load_tasks(manifest)
    datamind_by_id = {task.task_id: task for task in datamind_tasks}
    missing_datamind_tasks = sorted(selected_ids["datamind_sql"] - datamind_by_id.keys())
    datamind_failures: list[dict[str, Any]] = []
    for task_id in sorted(selected_ids["datamind_sql"] - set(missing_datamind_tasks)):
        task = datamind_by_id[task_id]
        try:
            datamind_store.resolve(task.database_relpath)
            gold_path = datamind_store.resolve(task.gold_relpath)
            grade = datamind.grade_result(task, gold_path, datamind_store)
            if grade.get("reward") != 1.0:
                datamind_failures.append({"task_id": task_id, "grade": grade})
        except Exception as exc:
            datamind_failures.append({"task_id": task_id, "error": str(exc)})
    if missing_datamind_tasks or datamind_failures:
        blockers.append("DataMind artifacts or ground-truth self-scores are incomplete")

    environments = {
        "dare": _environment_fingerprint(
            args.dare_python, expected_python=(3, 10), packages=DARE_PACKAGES
        ),
        "general": _environment_fingerprint(
            args.general_python, expected_python=(3, 12), packages=GENERAL_PACKAGES
        ),
    }
    for role, environment in environments.items():
        if not environment.get("passed"):
            blockers.append(f"{role} sandbox environment does not match its local contract")
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap:
        blockers.append("bubblewrap is unavailable")

    return {
        "schema_version": 1,
        "status": "passed" if not blockers else "failed",
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "tasks": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
        },
        "repository": _repository_fingerprint(Path(__file__).resolve().parents[2]),
        "isolation": {"local_backend": "bubblewrap", "executable": bubblewrap},
        "environments": environments,
        "sources": {
            "dare_bench": {
                "selected": len(selected_ids["dare_bench"]),
                "missing_task_specs": missing_dare_tasks,
                "ground_truth_failures": dare_failures,
            },
            "data_agent_rl": {
                "selected": len(selected_ids["data_agent_rl"]),
                "unique_dataset_prefixes": len(
                    {
                        agent_by_id[task_id].bucket_prefix
                        for task_id in selected_ids["data_agent_rl"] - set(missing_agent_tasks)
                    }
                ),
                "missing_task_specs": missing_agent_tasks,
                "missing_cache_prefix_count": len(missing_cache_prefixes),
                "missing_cache_prefix_examples": missing_cache_prefixes[:20],
            },
            "datamind_sql": {
                "selected": len(selected_ids["datamind_sql"]),
                "missing_task_specs": missing_datamind_tasks,
                "ground_truth_failures": datamind_failures,
            },
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dare-manifest", type=Path, required=True)
    parser.add_argument("--dare-databases-dir", type=Path, required=True)
    parser.add_argument("--data-agent-root", type=Path, required=True)
    parser.add_argument("--data-agent-cache", type=Path, required=True)
    parser.add_argument("--datamind-root", type=Path, required=True)
    parser.add_argument("--dare-python", type=Path, required=True)
    parser.add_argument("--general-python", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_preflight(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
