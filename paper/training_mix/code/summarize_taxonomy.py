#!/usr/bin/env python3
"""Audit taxonomy annotations and write paper-facing summary artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from paper.training_mix.code.classify_taxonomy_deepseek import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_TAXONOMY,
    load_jsonl,
    load_taxonomy,
    task_evidence_text,
    validate_decision,
)

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = PAPER_ROOT / "artifacts" / "audits"


def jaccard_mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.mean(float(row["agreement"][field]) for row in rows)


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    summary = {
        "tasks": total,
        "primary_match_rate": sum(row["agreement"]["primary_match"] for row in rows) / total,
        "secondary_jaccard_mean": jaccard_mean(rows, "secondary_jaccard"),
        "operations_jaccard_mean": jaccard_mean(rows, "operations_jaccard"),
        "adjudicated": sum(row["adjudicated"] for row in rows),
        "needs_review": sum(row["needs_review"] for row in rows),
        "unresolved": sum(row["final"]["primary_task"] == "unresolved" for row in rows),
        "confidence_mean": statistics.mean(row["final"]["confidence"] for row in rows),
        "primary_counts": dict(Counter(row["final"]["primary_task"] for row in rows).most_common()),
    }
    official_rows = [row for row in rows if row.get("official_primary_task")]
    if official_rows:
        summary["model_official_disagreements"] = sum(
            row["model_primary_before_official_override"] != row["official_primary_task"]
            for row in official_rows
        )
    return summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit(
    manifest_path: Path, annotations_path: Path, taxonomy_path: Path, output_dir: Path
) -> None:
    manifest = load_jsonl(manifest_path)
    annotations = load_jsonl(annotations_path)
    taxonomy = load_taxonomy(taxonomy_path)
    manifest_by_id = {str(row["task_id"]): row for row in manifest}
    annotations_by_id = {str(row["task_id"]): row for row in annotations}

    if len(manifest) != 2000 or len(manifest_by_id) != 2000:
        raise ValueError("Manifest must contain exactly 2,000 unique task IDs")
    if len(annotations) != 2000 or len(annotations_by_id) != 2000:
        raise ValueError("Annotations must contain exactly 2,000 unique task IDs")
    if set(manifest_by_id) != set(annotations_by_id):
        raise ValueError("Manifest and annotation task IDs do not match")

    invalid: list[dict[str, str]] = []
    for task_id, annotation in annotations_by_id.items():
        task = manifest_by_id[task_id]
        question = str(task["question"])
        expected_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        try:
            if annotation.get("status") != "ok":
                raise ValueError(f"status={annotation.get('status')}")
            if annotation.get("question_sha256") != expected_hash:
                raise ValueError("question hash mismatch")
            validate_decision(annotation["final"], taxonomy, task_evidence_text(task, None))
        except Exception as exc:
            invalid.append({"task_id": task_id, "error": str(exc)})
    if invalid:
        raise ValueError(f"Found {len(invalid)} invalid annotations; first={invalid[0]}")

    source_names = sorted({str(row["source"]) for row in annotations})
    primary_counts = Counter(row["final"]["primary_task"] for row in annotations)
    review_rows = [row for row in annotations if row["needs_review"]]
    audit_result = {
        "status": "passed",
        "taxonomy_version": taxonomy.version,
        "tasks": len(annotations),
        "unique_task_ids": len(annotations_by_id),
        "question_hash_failures": 0,
        "structured_validation_failures": 0,
        "evidence_validation_failures": 0,
        "primary_match_rate": sum(row["agreement"]["primary_match"] for row in annotations)
        / len(annotations),
        "adjudicated": sum(row["adjudicated"] for row in annotations),
        "needs_review": len(review_rows),
        "unresolved": primary_counts["unresolved"],
        "confidence_buckets": {
            "ge_0_90": sum(row["final"]["confidence"] >= 0.90 for row in annotations),
            "ge_0_75_lt_0_90": sum(
                0.75 <= row["final"]["confidence"] < 0.90 for row in annotations
            ),
            "lt_0_75": sum(row["final"]["confidence"] < 0.75 for row in annotations),
        },
        "primary_counts": dict(primary_counts.most_common()),
        "by_source": {
            source: source_summary([row for row in annotations if str(row["source"]) == source])
            for source in source_names
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "taxonomy_audit.json", audit_result)
    with (output_dir / "taxonomy_review_queue.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "source",
                "primary_task",
                "confidence",
                "unresolved",
                "adjudicated",
                "question",
            ],
        )
        writer.writeheader()
        for row in sorted(review_rows, key=lambda item: (item["source"], item["task_id"])):
            writer.writerow(
                {
                    "task_id": row["task_id"],
                    "source": row["source"],
                    "primary_task": row["final"]["primary_task"],
                    "confidence": row["final"]["confidence"],
                    "unresolved": row["final"]["primary_task"] == "unresolved",
                    "adjudicated": row["adjudicated"],
                    "question": manifest_by_id[row["task_id"]]["question"],
                }
            )
    print(json.dumps(audit_result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    audit(
        args.manifest.expanduser().resolve(),
        args.annotations.expanduser().resolve(),
        args.taxonomy.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
