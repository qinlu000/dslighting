"""Materialize only the files referenced by the frozen 2k training mixture."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from experiments.common import read_jsonl, sha256_file, write_json, write_jsonl
from experiments.dare_bench.runner import load_tasks as load_dare_tasks
from experiments.data_agent_rl.runner import BucketDataStore
from experiments.data_agent_rl.runner import load_tasks as load_agent_tasks


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(args: argparse.Namespace) -> dict:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Bundle output already exists: {output}")
    output.mkdir(parents=True)

    mix_path = args.manifest.expanduser().resolve()
    mix_rows = read_jsonl(mix_path)
    selected = {
        source: {str(row["task_id"]) for row in mix_rows if row["source"] == source}
        for source in ("dare_bench", "data_agent_rl", "datamind_sql")
    }
    write_jsonl(output / "manifests" / "train_full.jsonl", mix_rows)

    dare_source_root = args.dare_databases_dir.expanduser().resolve()
    dare_tasks = [
        task
        for task in load_dare_tasks(args.dare_manifest)
        if task.task_id in selected["dare_bench"]
    ]
    if len(dare_tasks) != len(selected["dare_bench"]):
        raise ValueError("DARE selection is incomplete")
    write_jsonl(
        output / "dare" / "tasks.jsonl",
        [dict(task.upstream_record) for task in dare_tasks],
    )
    dare_relpaths = {
        relative
        for task in dare_tasks
        for relative in (*task.needed_file_members, task.ground_truth_relpath)
    }
    for relative in sorted(dare_relpaths):
        _link_or_copy(dare_source_root / relative, output / "dare" / "databases" / relative)

    agent_source_root = args.data_agent_root.expanduser().resolve()
    all_agent_tasks = load_agent_tasks(agent_source_root)
    agent_tasks = [task for task in all_agent_tasks if task.task_id in selected["data_agent_rl"]]
    if len(agent_tasks) != len(selected["data_agent_rl"]):
        raise ValueError("Data Agent selection is incomplete")
    source_frame = pd.read_parquet(agent_source_root / "manifest.parquet")
    selected_frame = source_frame[source_frame["task_dir"].isin(selected["data_agent_rl"])]
    if len(selected_frame) != len(agent_tasks):
        raise ValueError("Data Agent manifest filtering is incomplete")
    agent_dataset = output / "data_agent" / "dataset"
    agent_dataset.mkdir(parents=True)
    selected_frame.to_parquet(agent_dataset / "manifest.parquet", index=False)
    for task in agent_tasks:
        source_toml = agent_source_root / "tasks" / task.task_id / "task.toml"
        _link_or_copy(
            source_toml,
            agent_dataset / "tasks" / task.task_id / "task.toml",
        )

    source_cache = BucketDataStore(args.data_agent_cache, offline=True)
    target_cache = output / "data_agent" / "cache"
    by_prefix: dict[str, list] = {}
    for task in agent_tasks:
        if source_cache.cached_dataset(task) is None:
            raise ValueError(f"Data Agent cache is incomplete for {task.bucket_prefix}")
        by_prefix.setdefault(task.bucket_prefix, []).append(task)
    for prefix, tasks in sorted(by_prefix.items()):
        marker = source_cache.cache_root / ".complete" / f"{prefix}.json"
        names = json.loads(marker.read_text(encoding="utf-8"))["files"]
        required = {Path(value).name for task in tasks for value in task.files_used}
        if not required.issubset(set(names)):
            raise ValueError(f"Data Agent marker omits required files for {prefix}")
        selected_names = sorted(required)
        for name in selected_names:
            _link_or_copy(
                source_cache.cache_root / prefix / name,
                target_cache / prefix / name,
            )
        target_marker = target_cache / ".complete" / f"{prefix}.json"
        write_json(
            target_marker,
            {"bucket_id": source_cache.bucket_id, "prefix": prefix, "files": selected_names},
        )

    datamind_source_root = args.datamind_root.expanduser().resolve()
    datamind_rows = [row for row in mix_rows if row["source"] == "datamind_sql"]
    datamind_relpaths = {
        relative
        for row in datamind_rows
        for relative in (
            row["input_ref"]["ref"],
            row["evaluator"]["ground_truth_ref"],
        )
    }
    for relative in sorted(datamind_relpaths):
        _link_or_copy(
            datamind_source_root / relative,
            output / "datamind" / relative,
        )

    files = sorted(path for path in output.rglob("*") if path.is_file())
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    report = {
        "schema_version": 1,
        "tasks": len(mix_rows),
        "source_counts": {source: len(ids) for source, ids in selected.items()},
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "manifest_sha256": sha256_file(output / "manifests" / "train_full.jsonl"),
        "entries": entries,
    }
    write_json(output / "bundle_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dare-manifest", type=Path, required=True)
    parser.add_argument("--dare-databases-dir", type=Path, required=True)
    parser.add_argument("--data-agent-root", type=Path, required=True)
    parser.add_argument("--data-agent-cache", type=Path, required=True)
    parser.add_argument("--datamind-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = materialize(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "entries"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
