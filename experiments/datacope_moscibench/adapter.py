"""Convert DSLighting MoSciBench runs into DataCOPE inputs."""

from __future__ import annotations

import csv
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

if __package__:
    from .split import MoSciSplit, load_split
else:
    from split import MoSciSplit, load_split


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc


def _require_empty(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"{label} must be empty or new: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def normalize_answer(value: str) -> str:
    text = str(value or "").strip()
    tagged = re.fullmatch(r"@answer\[(.*)]", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        text = tagged.group(1).strip()
    braced = re.findall(r"\{([^{}]+)\}", text)
    if braced:
        text = braced[-1].strip()
    compact = " ".join(text.split())
    lower = compact.lower()
    if lower in {"true", "false"}:
        return f"bool:{lower}"
    try:
        number = Decimal(compact.replace("%", ""))
        if number.is_finite():
            return f"number:{format(number.normalize(), 'f')}"
    except InvalidOperation:
        pass
    return f"text:{lower}" if lower else "text:<missing>"


def _submission(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "prediction": "MISSING", "rows": []}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"status": "invalid", "prediction": "INVALID", "rows": [], "error": str(exc)}
    if len(rows) != 1 or "answer" not in rows[0]:
        return {
            "status": "invalid",
            "prediction": "INVALID",
            "rows": rows,
            "error": "Expected one submission row with an answer column",
        }
    return {
        "status": "valid",
        "prediction": normalize_answer(rows[0]["answer"]),
        "rows": rows,
    }


def prepare_public_data(
    data_root: Path,
    split_manifest: Path,
    output_dir: Path,
) -> Path:
    """Expose only explore descriptions and prepared/public data to SkillManager."""

    root = Path(data_root).expanduser().resolve()
    split = load_split(split_manifest)
    split.verify_source(root)
    output = _require_empty(output_dir, "Public data directory")
    task_root = output / "tasks"
    task_root.mkdir()
    tasks: list[dict[str, str]] = []
    for task_id in split.task_ids("explore"):
        source = root / task_id
        description = (source / "description.md").read_text(encoding="utf-8")
        tasks.append({"query_id": task_id, "description": description})
        os.symlink(
            (source / "prepared" / "public").resolve(),
            task_root / task_id,
            target_is_directory=True,
        )
    (output / "tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def validate_public_data(data_dir: Path, expected_query_ids: set[str]) -> tuple[Path, ...]:
    root = Path(data_dir).expanduser().resolve()
    tasks = _read_json(root / "tasks.json")
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("Public tasks.json must contain a list of objects")
    if {str(task.get("query_id", "")) for task in tasks} != expected_query_ids:
        raise ValueError("Public task set differs from explore predictions")

    targets = []
    for task_id in sorted(expected_query_ids):
        link = root / "tasks" / task_id
        if not link.is_symlink():
            raise ValueError(f"Expected a prepared/public symlink: {link}")
        target = link.resolve()
        if not target.is_dir() or target.name != "public":
            raise ValueError(f"Invalid public-data target: {target}")
        targets.append(target)
    return tuple(targets)


def _metadata_files(workspace_root: Path) -> list[Path]:
    return sorted(workspace_root.glob("*/artifacts/telemetry/run_metadata.json"))


def _submission_path(metadata: dict[str, Any], workspace_dir: Path) -> Path:
    summary = metadata.get("summary") or {}
    result = summary.get("result") or {}
    task_context = metadata.get("task_context") or {}
    payload = (metadata.get("task") or {}).get("payload") or {}
    candidates = [
        result.get("submission_path") if isinstance(result, dict) else None,
        task_context.get("expected_output_path"),
        payload.get("output_submission_path"),
    ]
    value = next((str(item).strip() for item in candidates if str(item or "").strip()), "")
    if not value:
        return workspace_dir / "__missing_submission__.csv"
    path = Path(value).expanduser()
    return path if path.is_absolute() else workspace_dir / path


def export_workspace_sample(
    workspace_root: Path,
    split_manifest: Path,
    output_dir: Path,
) -> list[Path]:
    workspace = Path(workspace_root).expanduser().resolve()
    split: MoSciSplit = load_split(split_manifest)
    expected = set(split.task_ids("explore"))
    output = _require_empty(output_dir, "Prediction output directory")

    records: dict[str, tuple[dict[str, Any], Path]] = {}
    for metadata_file in _metadata_files(workspace):
        metadata = _read_json(metadata_file)
        task_id = str((metadata.get("task") or {}).get("task_id") or "")
        if task_id in expected:
            records[task_id] = (metadata, metadata_file)
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise ValueError(f"Workspace does not contain the frozen explore task set: {missing[:5]}")

    written: list[Path] = []
    for index, task_id in enumerate(sorted(expected)):
        metadata, metadata_file = records[task_id]
        messages_file = metadata_file.parents[1] / "messages.json"
        messages = _read_json(messages_file)
        if not isinstance(messages, list):
            raise ValueError(f"Expected ReAct messages in {messages_file}")
        task = metadata.get("task") or {}
        payload = task.get("payload") or {}
        workspace_dir = Path(metadata.get("workspace_dir") or metadata_file.parents[2])
        submission = _submission(_submission_path(metadata, workspace_dir))
        prediction = {
            "prediction": submission["prediction"],
            "query": str(payload.get("description") or task_id),
            "turns": sum(message.get("role") == "assistant" for message in messages),
            "conversation": messages,
            "submission": {key: value for key, value in submission.items() if key != "prediction"},
            "extra_info": {"query_id": task_id},
        }
        destination = output / f"prediction_{index:04d}.json"
        destination.write_text(
            json.dumps(prediction, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return written
