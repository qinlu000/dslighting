#!/usr/bin/env python3
"""Assemble Solver data with clean, completed Perception traces from many runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper.training_mix.code.sft import (
    DATASET_NAME,
    SOURCES,
    read_json,
    read_jsonl,
    runtime_artifacts,
    sft_record,
    strip_feedback_attempts,
    task_gate,
    task_score,
    token_counter,
    trajectory_hash,
    write_json,
    write_sft_dataset,
)
from paper.training_mix.code.trajectory_audit import audit_perception_record

OBSERVATION_ERROR_PATTERNS = (
    "traceback",
    "exception",
    "execution failed",
    "timed out",
    "timeout",
    "syntaxerror",
    "memoryerror",
    "killed",
)


def _observation_errors(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    observations = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith("<Observation>")
    ).lower()
    return [pattern for pattern in OBSERVATION_ERROR_PATTERNS if pattern in observations]


def _solver_records(base_output_dir: Path) -> list[dict[str, Any]]:
    path = base_output_dir / "clean" / "solver_sft_clean.json"
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing base Solver records: {path}")
    if any(row.get("metadata", {}).get("role") != "solver" for row in records):
        raise ValueError("Base Solver file contains a non-Solver record")
    return records


def assemble(
    *,
    base_output_dir: Path,
    candidates_path: Path,
    run_roots: Sequence[Path],
    output_dir: Path,
    tokenizer_path: Path,
    cutoff_len: int,
    seed: int,
) -> dict[str, Any]:
    candidates = {
        (str(row["source"]), str(row["task_id"])): row for row in read_jsonl(candidates_path)
    }
    solver_records = _solver_records(base_output_dir)
    count_tokens = token_counter(tokenizer_path)
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []
    observed_results = Counter()

    for run_index, run_root in enumerate(run_roots):
        repeat_root = run_root / "repetitions" / "repeat_00"
        for source in SOURCES:
            for result_path in sorted((repeat_root / source).glob("*/result.json")):
                result = read_json(result_path)
                task_id = str(result.get("task_id") or "")
                key = (source, task_id)
                candidate = candidates.get(key)
                if candidate is None:
                    continue
                observed_results[run_root.name] += 1
                score = task_score(source, result)
                _, task_gate_error = task_gate(source, result)
                task_success = task_gate_error is None
                workspace, _, runtime_error = runtime_artifacts(result)
                sessions_path = (
                    workspace / "artifacts" / "perception_sessions.json"
                    if workspace is not None
                    else None
                )
                sessions = (
                    read_json(sessions_path)
                    if sessions_path is not None and sessions_path.is_file()
                    else []
                )
                if not isinstance(sessions, list):
                    sessions = []

                for session_index, session in enumerate(sessions):
                    reasons: list[str] = []
                    raw_messages = session.get("messages") if isinstance(session, Mapping) else None
                    messages = None
                    feedback_removed = 0
                    if isinstance(raw_messages, list):
                        messages, feedback_removed = strip_feedback_attempts(raw_messages)
                    audit = audit_perception_record(
                        {
                            "messages": messages,
                            "metadata": {
                                "session_status": (
                                    session.get("status") if isinstance(session, Mapping) else None
                                )
                            },
                        },
                        token_counter=count_tokens,
                        cutoff_len=cutoff_len,
                    )
                    reasons.extend(audit.reasons)
                    if runtime_error is not None:
                        reasons.append(runtime_error)
                    error_patterns = (
                        _observation_errors(messages) if isinstance(messages, list) else []
                    )
                    if error_patterns:
                        reasons.append("observation_execution_error")

                    session_hash = None
                    if not reasons and isinstance(messages, list) and score is not None:
                        session_hash = trajectory_hash(messages)
                        record = sft_record(
                            messages=messages,
                            role="perception",
                            candidate=candidate,
                            result=result,
                            score=score,
                            token_count=int(audit.token_count or 0),
                            trajectory_hash=session_hash,
                            feedback_pairs_removed=feedback_removed,
                            session=session,
                            session_index=session_index,
                        )
                        record["metadata"].update(
                            {
                                "solver_task_success": task_success,
                                "solver_task_gate_error": task_gate_error,
                                "acceptance_policy": (
                                    "completed_clean_perception_independent_of_solver_outcome"
                                ),
                                "run_root": str(run_root),
                                "run_index": run_index,
                            }
                        )
                        eligible[key].append(record)

                    audit_rows.append(
                        {
                            "source": source,
                            "task_id": task_id,
                            "run_root": str(run_root),
                            "session_index": session_index,
                            "session_status": (
                                session.get("status") if isinstance(session, Mapping) else None
                            ),
                            "solver_task_success": task_success,
                            "task_score": score,
                            "accepted_by_session_gate": not reasons,
                            "reasons": sorted(set(reasons)),
                            "observation_error_patterns": error_patterns,
                            "token_count": audit.token_count,
                            "trajectory_sha256": session_hash,
                        }
                    )

    perception_records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for key in sorted(eligible):
        choices = sorted(
            eligible[key],
            key=lambda row: (
                not bool(row["metadata"]["solver_task_success"]),
                int(row["metadata"]["feedback_pairs_removed"]),
                int(row["metadata"]["chat_template_tokens"]),
                int(row["metadata"]["run_index"]),
                str(row["metadata"]["trajectory_sha256"]),
            ),
        )
        selected = choices[0]
        selected_hash = str(selected["metadata"]["trajectory_sha256"])
        if selected_hash in seen_hashes:
            continue
        seen_hashes.add(selected_hash)
        perception_records.append(selected)

    joint_records, paths = write_sft_dataset(
        output_dir,
        solver_records=solver_records,
        perception_records=perception_records,
        manifest_rows=audit_rows,
        manifest_name="perception_audit_manifest.jsonl",
        seed=seed,
    )

    perception_by_source = Counter(row["metadata"]["source"] for row in perception_records)
    strict_supported = sum(
        bool(row["metadata"]["solver_task_success"]) for row in perception_records
    )
    reason_counts = Counter(reason for row in audit_rows for reason in row["reasons"])
    all_hashes = [str(row["metadata"]["trajectory_sha256"]) for row in joint_records]
    summary = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "policy": {
            "solver": "unchanged strict task-success-gated records from the 900-task export",
            "perception": (
                "completed protocol-valid Perception with valid Python, no execution-error "
                "observation, within cutoff, one record per task; Solver outcome is advisory"
            ),
            "deduplication": "unique trajectory and at most one Perception record per task",
        },
        "inputs": {
            "base_output_dir": str(base_output_dir),
            "candidates": str(candidates_path),
            "run_roots": [str(path) for path in run_roots],
            "observed_results_by_run": dict(observed_results),
        },
        "accepted": {
            "solver": len(solver_records),
            "perception": len(perception_records),
            "perception_with_successful_solver_outcome": strict_supported,
            "perception_independent_of_solver_outcome": len(perception_records) - strict_supported,
            "total": len(joint_records),
            "perception_by_source": dict(sorted(perception_by_source.items())),
        },
        "rejected_session_reason_counts": dict(sorted(reason_counts.items())),
        "validation": {
            "unique_trajectories": len(all_hashes) == len(set(all_hashes)),
            "unique_perception_tasks": len(perception_records) == len(eligible),
            "all_within_cutoff": all(
                int(row["metadata"]["chat_template_tokens"]) <= cutoff_len for row in joint_records
            ),
            "all_perception_completed": all(
                row["metadata"].get("session_status") == "completed" for row in perception_records
            ),
            "max_tokens": max(
                int(row["metadata"]["chat_template_tokens"]) for row in joint_records
            ),
        },
        "trainable": {
            "format": "LLaMA-Factory ShareGPT messages",
            "records": len(joint_records),
            "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
        },
        "outputs": {
            "solver": str(paths["solver"]),
            "perception": str(paths["perception"]),
            "joint": str(paths["joint"]),
            "manifest": str(paths["manifest"]),
            "train": str(paths["train"]),
            "dataset_info": str(paths["dataset_info"]),
            "summary": str(paths["summary"]),
        },
    }
    if not all(summary["validation"].values()):
        raise AssertionError(f"Final dataset validation failed: {summary['validation']}")
    write_json(paths["summary"], summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=24_576)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = assemble(
        base_output_dir=args.base_output_dir.expanduser().resolve(),
        candidates_path=args.candidates.expanduser().resolve(),
        run_roots=[path.expanduser().resolve() for path in args.run_root],
        output_dir=args.output_dir.expanduser().resolve(),
        tokenizer_path=args.tokenizer.expanduser().resolve(),
        cutoff_len=args.cutoff_len,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
