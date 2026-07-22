"""Convert one DSLighting DABench run into DataCOPE prediction files."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

PUBLIC_DATA_FILES = frozenset({"train.csv", "sample_submission.csv"})
_TASK_ID_PATTERN = re.compile(r"dabench-[A-Za-z0-9][A-Za-z0-9._-]*")


def submission_signature(path: Path) -> dict[str, Any]:
    """Return a stable signature for a CSV submission.

    The signature deliberately checks only the observable artifact. It does not
    call a grader or compare against a sample submission.
    """

    path = Path(path)
    if not path.is_file():
        return {
            "status": "missing",
            "prediction": "MISSING",
            "sha256": None,
            "rows": [],
        }

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        return {
            "status": "invalid",
            "prediction": "INVALID",
            "sha256": None,
            "rows": [],
            "error": str(exc),
        }

    if len(rows) < 2 or not rows[0]:
        return {
            "status": "invalid",
            "prediction": "INVALID",
            "sha256": None,
            "rows": rows,
            "error": "CSV must contain a header and at least one data row",
        }

    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        return {
            "status": "invalid",
            "prediction": "INVALID",
            "sha256": None,
            "rows": rows,
            "error": "CSV rows have different column counts",
        }

    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "status": "valid",
        "prediction": f"VALID:{digest}",
        "sha256": digest,
        "rows": rows,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def _metadata_files(workspace_root: Path) -> list[Path]:
    direct = workspace_root / "artifacts" / "telemetry" / "run_metadata.json"
    if direct.is_file():
        return [direct]
    return sorted(workspace_root.glob("*/artifacts/telemetry/run_metadata.json"))


def _submission_path(metadata: dict[str, Any], workspace_dir: Path) -> Path:
    summary = metadata.get("summary") or {}
    result = summary.get("result") or {}
    task_context = metadata.get("task_context") or {}
    task = metadata.get("task") or {}
    payload = task.get("payload") or {}

    candidates = [
        result.get("submission_path") if isinstance(result, dict) else None,
        task_context.get("expected_output_path"),
        payload.get("output_submission_path"),
    ]
    raw_path = next((str(value).strip() for value in candidates if str(value or "").strip()), "")
    if not raw_path:
        return workspace_dir / "__missing_submission__.csv"

    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else workspace_dir / path


def _load_messages(metadata_file: Path) -> list[dict[str, Any]]:
    messages_file = metadata_file.parents[1] / "messages.json"
    if not messages_file.is_file():
        raise ValueError(
            f"Missing ReAct trajectory {messages_file}; this PoC does not export "
            "AIDE search trees because they may contain grader-derived feedback"
        )

    messages = _read_json(messages_file)
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError(f"Expected a list of messages in {messages_file}")
    return messages


def export_workspace_run(workspace_root: Path, output_dir: Path) -> list[Path]:
    """Export one stochastic DABench run as one DataCOPE sample directory.

    ``workspace_root`` should be the dedicated DSLighting workspace root for a
    single benchmark run. Repeated runs are exported into sibling directories
    such as ``predictions/run_0`` and ``predictions/run_1``.
    """

    workspace_root = Path(workspace_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    metadata_files = _metadata_files(workspace_root)
    if not metadata_files:
        raise ValueError(f"No run_metadata.json files found under {workspace_root}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_task: dict[str, tuple[dict[str, Any], Path]] = {}
    for metadata_file in metadata_files:
        metadata = _read_json(metadata_file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a JSON object in {metadata_file}")

        task = metadata.get("task") or {}
        task_id = str(task.get("task_id") or "").strip()
        if not task_id.startswith("dabench-"):
            continue

        previous = records_by_task.get(task_id)
        if previous is None:
            records_by_task[task_id] = (metadata, metadata_file)
            continue

        previous_metadata, previous_file = previous
        previous_timeline = previous_metadata.get("timeline") or {}
        current_timeline = metadata.get("timeline") or {}
        previous_key = (
            str(previous_timeline.get("ended_at_utc") or ""),
            str(previous_file),
        )
        current_key = (
            str(current_timeline.get("ended_at_utc") or ""),
            str(metadata_file),
        )
        if current_key > previous_key:
            records_by_task[task_id] = (metadata, metadata_file)

    if not records_by_task:
        raise ValueError(f"No DABench task workspaces found under {workspace_root}")

    written: list[Path] = []
    for index, (task_id, record) in enumerate(sorted(records_by_task.items())):
        metadata, metadata_file = record
        task = metadata.get("task") or {}
        payload = task.get("payload") or {}
        workspace_dir = Path(metadata.get("workspace_dir") or metadata_file.parents[2])
        messages = _load_messages(metadata_file)
        signature = submission_signature(_submission_path(metadata, workspace_dir))

        prediction = {
            "prediction": signature["prediction"],
            "query": str(payload.get("description") or task_id),
            "turns": sum(message.get("role") == "assistant" for message in messages),
            "conversation": messages,
            "submission": {key: value for key, value in signature.items() if key != "prediction"},
            "extra_info": {"query_id": task_id},
        }
        destination = output_dir / f"prediction_{index:04d}.json"
        destination.write_text(
            json.dumps(prediction, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(destination)

    return written


def prepare_public_data(
    source_root: Path,
    output_dir: Path,
    task_ids: Iterable[str],
) -> list[Path]:
    """Copy the requested tasks' public files into an explore-only view."""

    source_root = Path(source_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"DABench source root does not exist: {source_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")

    requested = [str(task_id).strip() for task_id in task_ids]
    if not requested:
        raise ValueError("At least one explore task ID is required")
    if len(requested) != len(set(requested)):
        raise ValueError("Explore task IDs must be unique")
    invalid = sorted(task_id for task_id in requested if not _TASK_ID_PATTERN.fullmatch(task_id))
    if invalid:
        raise ValueError(f"Invalid DABench task IDs: {invalid[:5]}")

    public_dirs: list[tuple[str, Path]] = []
    for task_id in sorted(requested):
        task_dir = source_root / task_id
        public_dir = task_dir / "prepared" / "public"
        if task_dir.is_symlink() or public_dir.is_symlink():
            raise ValueError(f"DABench source task must not use symlinks: {task_id}")
        if not public_dir.is_dir():
            raise ValueError(f"Missing prepared/public directory for {task_id}")

        entries = list(public_dir.iterdir())
        names = {entry.name for entry in entries}
        if names != PUBLIC_DATA_FILES:
            raise ValueError(
                f"Unexpected public files for {task_id}; "
                f"expected={sorted(PUBLIC_DATA_FILES)}, actual={sorted(names)}"
            )
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError(f"Public data files must be regular files for {task_id}")
        public_dirs.append((task_id, public_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_id, public_dir in public_dirs:
        destination = output_dir / task_id
        destination.mkdir()
        for filename in sorted(PUBLIC_DATA_FILES):
            shutil.copy2(public_dir / filename, destination / filename)
        written.append(destination)
    return written
