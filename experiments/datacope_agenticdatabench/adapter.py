"""Convert AgenticDataBench DSLighting runs into DataCOPE inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

if __package__:
    from .split import (
        AgenticDataBenchSplit,
        TaskRecord,
        load_split,
        load_task_records,
        required_output_names,
    )
else:
    from split import (
        AgenticDataBenchSplit,
        TaskRecord,
        load_split,
        load_task_records,
        required_output_names,
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def _empty_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"{label} must be empty or new: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _task_map(tasks_file: Path) -> dict[str, TaskRecord]:
    return {task.task_id: task for task in load_task_records(tasks_file)}


def _domain_dir(dataset_root: Path, domain: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    target = (root / domain).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"AgenticDataBench domain escapes dataset root: {domain}") from exc
    if not target.is_dir():
        raise ValueError(f"AgenticDataBench public domain does not exist: {target}")
    return target


def prepare_public_data(
    tasks_file: Path,
    dataset_root: Path,
    split_manifest: Path,
    output_dir: Path,
) -> Path:
    """Expose only explore questions and their public domain directories."""

    split = load_split(split_manifest)
    split.verify_source(tasks_file)
    tasks = _task_map(tasks_file)
    output = _empty_directory(output_dir, "Public data directory")
    links = output / "tasks"
    links.mkdir()
    records: list[dict[str, Any]] = []
    for task_id in split.explore_task_ids:
        task = tasks[task_id]
        target = _domain_dir(dataset_root, task.domain)
        os.symlink(target, links / task_id, target_is_directory=True)
        records.append(
            {
                "query_id": task.task_id,
                "question": task.question,
                "domain": task.domain,
                "required_outputs": list(required_output_names(task)),
            }
        )
    (output / "tasks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def validate_public_data(
    data_dir: Path,
    expected_query_ids: set[str],
) -> tuple[Path, ...]:
    root = Path(data_dir).expanduser().resolve()
    records = _read_json(root / "tasks.json")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("AgenticDataBench public tasks.json must be a list of objects")
    query_ids = [str(record.get("query_id") or "") for record in records]
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != expected_query_ids:
        raise ValueError("AgenticDataBench public task set differs from predictions")

    links = root / "tasks"
    if not links.is_dir() or {path.name for path in links.iterdir()} != expected_query_ids:
        raise ValueError("AgenticDataBench public data links differ from predictions")
    targets: set[Path] = set()
    for task_id in sorted(expected_query_ids):
        link = links / task_id
        if not link.is_symlink():
            raise ValueError(f"Expected a public domain symlink: {link}")
        target = link.resolve()
        if not target.is_dir():
            raise ValueError(f"Invalid AgenticDataBench public domain: {target}")
        targets.add(target)
    return tuple(sorted(targets))


def output_signature(task: TaskRecord, task_output_dir: Path) -> dict[str, Any]:
    root = Path(task_output_dir).expanduser().resolve()
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in required_output_names(task):
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        files.append({"name": name, "size": path.stat().st_size, "sha256": digest})

    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    manifest_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    if missing:
        prediction = "MISSING:" + ",".join(sorted(missing))
        status = "missing"
    else:
        prediction = f"VALID:{manifest_sha256}"
        status = "valid"
    return {
        "status": status,
        "manifest_sha256": manifest_sha256,
        "files": files,
        "missing_files": missing,
        "prediction": prediction,
    }


def export_sample_run(
    run_output_dir: Path,
    tasks_file: Path,
    split_manifest: Path,
    output_dir: Path,
) -> list[Path]:
    """Export one stochastic explore run without grader or gold information."""

    run_root = Path(run_output_dir).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"AgenticDataBench run output does not exist: {run_root}")
    split: AgenticDataBenchSplit = load_split(split_manifest)
    split.verify_source(tasks_file)
    tasks = _task_map(tasks_file)
    output = _empty_directory(output_dir, "Prediction output directory")

    written: list[Path] = []
    for index, task_id in enumerate(split.explore_task_ids):
        task = tasks[task_id]
        task_output = run_root / task_id
        result_path = task_output / "dabench" / "result.json"
        result = _read_json(result_path)
        trajectory = result.get("trajectory") if isinstance(result, dict) else None
        if not isinstance(trajectory, list) or not all(
            isinstance(message, dict) for message in trajectory
        ):
            raise ValueError(f"Missing DSLighting trajectory in {result_path}")
        signature = output_signature(task, task_output)
        prediction = {
            "prediction": signature["prediction"],
            "query": task.question,
            "turns": sum(message.get("role") == "assistant" for message in trajectory),
            "conversation": trajectory,
            "submission": {
                key: value for key, value in signature.items() if key != "prediction"
            },
            "extra_info": {"query_id": task.task_id, "domain": task.domain},
        }
        destination = output / f"prediction_{index:04d}.json"
        destination.write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return written
