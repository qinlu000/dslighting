#!/usr/bin/env python3
"""Build leakage-aware DARE-Bench train manifests from the unextracted archive."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

FIXED_GT_NAME = {
    ("classification", "v2"): "ground_truth.csv",
    ("regression", "v2"): "ground_truth.csv",
    ("time_series_analysis", "v1"): "ground_truth_v1.csv",
    ("time_series_analysis", "v2"): "ground_truth_v2.csv",
}

METRIC = {
    ("classification", "v2"): "macro_f1",
    ("regression", "v2"): "clipped_r2",
    ("time_series_analysis", "v1"): "clipped_r2",
    ("time_series_analysis", "v2"): "clipped_r2",
}

EXPECTED_BASE_COUNTS = {
    "classification": 807,
    "regression": 649,
    "time_series_analysis": 681,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def ground_truth_header(
    archive: zipfile.ZipFile, member: str
) -> tuple[list[str], int]:
    info = archive.getinfo(member)
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        header = next(csv.reader(text), [])
    return header, info.file_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-list", type=Path, required=True)
    parser.add_argument("--databases-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = json.loads(args.question_list.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise TypeError("question_list.json must contain a JSON list")

    fixed_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen_base_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    base_counts: collections.Counter[str] = collections.Counter()
    fixed_counts: collections.Counter[str] = collections.Counter()
    excluded_counts: collections.Counter[str] = collections.Counter()
    metric_counts: collections.Counter[str] = collections.Counter()
    format_counts: collections.Counter[str] = collections.Counter()
    nonportable_path_fields: collections.Counter[str] = collections.Counter()

    with zipfile.ZipFile(args.databases_zip) as archive:
        members = {item.filename: item for item in archive.infolist()}

        for index, item in enumerate(questions):
            base_id = item.get("file_path")
            task_type = item.get("task")
            if not isinstance(base_id, str) or not base_id:
                errors.append(f"question[{index}] has an invalid file_path")
                continue
            if base_id in seen_base_ids:
                errors.append(f"duplicate base task id: {base_id}")
                continue
            seen_base_ids.add(base_id)
            base_counts[str(task_type)] += 1

            if task_type not in EXPECTED_BASE_COUNTS:
                errors.append(f"{base_id}: unsupported task type {task_type!r}")
                continue

            metadata_member = f"{base_id}/verify/all_metadata.json"
            if metadata_member not in members:
                errors.append(f"{base_id}: missing {metadata_member}")
                continue
            metadata = json.loads(archive.read(metadata_member))
            question_meta = metadata.get("question", {})
            if question_meta.get("problem_type") != task_type:
                errors.append(
                    f"{base_id}: question task {task_type!r} != metadata problem_type "
                    f"{question_meta.get('problem_type')!r}"
                )

            targets = question_meta.get("target", [])
            if isinstance(targets, str):
                targets = [targets]
            if not targets or not all(isinstance(target, str) for target in targets):
                errors.append(f"{base_id}: invalid target metadata {targets!r}")

            save_file_type = str(question_meta.get("save_file_type"))
            format_counts[save_file_type] += 1
            for key, value in question_meta.items():
                if key.endswith("_file") and isinstance(value, str) and os.path.isabs(value):
                    nonportable_path_fields[key] += 1

            for version in ("v1", "v2"):
                task_id = f"{base_id}::{version}"
                if task_id in seen_task_ids:
                    errors.append(f"duplicate versioned task id: {task_id}")
                    continue
                seen_task_ids.add(task_id)

                question_key = f"question_{version}"
                question_text = item.get(question_key)
                if not isinstance(question_text, str) or not question_text.strip():
                    errors.append(f"{task_id}: missing {question_key}")

                needed_key = f"needed_files_{version}"
                list_needed = item.get(needed_key)
                meta_needed = question_meta.get(needed_key)
                if list_needed != meta_needed:
                    errors.append(
                        f"{task_id}: question-list and metadata needed_files differ: "
                        f"{list_needed!r} != {meta_needed!r}"
                    )
                needed_files = list(meta_needed or [])
                needed_members = [f"{base_id}/source/{name}" for name in needed_files]
                for member in needed_members:
                    info = members.get(member)
                    if info is None:
                        errors.append(f"{task_id}: missing source member {member}")
                    elif info.file_size == 0:
                        errors.append(f"{task_id}: empty source member {member}")

                common = {
                    "task_id": task_id,
                    "base_task_id": base_id,
                    "split": "train",
                    "version": version,
                    "task_type": task_type,
                    "agent_payload": {
                        "question": question_text,
                        "available_tools": item.get("available_tools", []),
                        "database_relpath": base_id,
                        "needed_files": needed_files,
                        "needed_file_members": needed_members,
                    },
                    "metadata": {
                        "all_metadata_relpath": metadata_member,
                        "target": targets,
                        "save_file_type": save_file_type,
                    },
                }

                gt_name = FIXED_GT_NAME.get((task_type, version))
                if gt_name is None:
                    record = {
                        **common,
                        "exclusion": {
                            "reason": "requires_reference_generation",
                            "expected_ground_truth_relpath": (
                                f"{base_id}/verify/simulated_pred_local.csv"
                            ),
                            "reference_model_type": question_meta.get("model_type"),
                            "reference_random_state": question_meta.get("random_state"),
                        },
                    }
                    excluded_records.append(record)
                    excluded_counts[f"{task_type}::{version}"] += 1
                    continue

                gt_member = f"{base_id}/verify/{gt_name}"
                if gt_member not in members:
                    errors.append(f"{task_id}: missing fixed ground truth {gt_member}")
                    header: list[str] = []
                    gt_size = 0
                else:
                    header, gt_size = ground_truth_header(archive, gt_member)
                    if gt_size == 0:
                        errors.append(f"{task_id}: empty fixed ground truth {gt_member}")
                    missing_targets = sorted(set(targets) - set(header))
                    if missing_targets:
                        errors.append(
                            f"{task_id}: ground truth header missing targets {missing_targets}: {header}"
                        )
                    id_columns = [column for column in header if column not in set(targets)]
                    if not id_columns:
                        errors.append(f"{task_id}: ground truth has no identifier columns: {header}")
                    if task_type in {"classification", "regression"} and "row_id" not in header:
                        errors.append(f"{task_id}: tabular ground truth lacks row_id: {header}")

                metric = METRIC[(task_type, version)]
                record = {
                    **common,
                    "grader": {
                        "type": "deterministic",
                        "metric": metric,
                        "ground_truth_relpath": gt_member,
                        "ground_truth_columns": header,
                        "ground_truth_size_bytes": gt_size,
                        "prediction_filename": "prediction.csv",
                    },
                }
                fixed_records.append(record)
                fixed_counts[f"{task_type}::{version}"] += 1
                metric_counts[metric] += 1

    if dict(base_counts) != EXPECTED_BASE_COUNTS:
        errors.append(
            f"base task counts differ from expected: {dict(base_counts)} != {EXPECTED_BASE_COUNTS}"
        )
    if len(fixed_records) != 2818:
        errors.append(f"fixed manifest count is {len(fixed_records)}, expected 2818")
    if len(excluded_records) != 1456:
        errors.append(f"excluded manifest count is {len(excluded_records)}, expected 1456")
    if len(seen_task_ids) != 4274:
        errors.append(f"versioned task ID count is {len(seen_task_ids)}, expected 4274")

    if nonportable_path_fields:
        warnings.append(
            "all_metadata.json contains author-machine absolute paths; manifests intentionally "
            "derive portable archive-relative paths instead"
        )

    summary = {
        "status": "failed" if errors else "passed",
        "source": {
            "question_list": str(args.question_list),
            "question_list_sha256": sha256_file(args.question_list),
            "databases_zip": str(args.databases_zip),
            "databases_zip_sha256": sha256_file(args.databases_zip),
        },
        "grain": {
            "base_task_key": "base_task_id",
            "training_sample_key": "task_id = base_task_id::version",
        },
        "counts": {
            "base_tasks": len(seen_base_ids),
            "versioned_samples": len(seen_task_ids),
            "fixed_ground_truth_samples": len(fixed_records),
            "excluded_reference_samples": len(excluded_records),
            "base_by_task_type": dict(sorted(base_counts.items())),
            "fixed_by_task_version": dict(sorted(fixed_counts.items())),
            "excluded_by_task_version": dict(sorted(excluded_counts.items())),
            "fixed_by_metric": dict(sorted(metric_counts.items())),
            "base_by_file_format": dict(sorted(format_counts.items())),
        },
        "quality": {
            "errors": errors,
            "warnings": warnings,
            "nonportable_metadata_path_fields": dict(
                sorted(nonportable_path_fields.items())
            ),
            "agent_leakage_control": (
                "Only agent_payload should be passed to the agent; grader and metadata are "
                "evaluator-only."
            ),
        },
    }

    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        raise SystemExit(f"validation failed with {len(errors)} error(s); no manifests written")

    fixed_path = args.output_dir / "train_fixed_gt.jsonl"
    excluded_path = args.output_dir / "excluded_reference_v1.jsonl"
    summary_path = args.output_dir / "manifest_summary.json"
    write_jsonl_atomic(fixed_path, fixed_records)
    write_jsonl_atomic(excluded_path, excluded_records)
    write_json_atomic(summary_path, summary)

    print(f"fixed={len(fixed_records)} path={fixed_path}")
    print(f"excluded={len(excluded_records)} path={excluded_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
