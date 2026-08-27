#!/usr/bin/env python3
"""Export correctness-gated Solver and Perception teacher traces for Joint SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper.training_mix.code.sft import (
    DATASET_NAME,
    SOURCES,
    TokenCounter,
    read_json,
    read_jsonl,
    result_index,
    runtime_artifacts,
    sft_record,
    strip_feedback_attempts,
    task_gate,
    token_counter,
    trajectory_hash,
    write_json,
    write_sft_dataset,
)
from paper.training_mix.code.trajectory_audit import (
    audit_perception_record,
    audit_solver_episode,
)


def _manifest_row(
    candidate: Mapping[str, Any],
    role: str,
    reasons: Sequence[str],
    score: float | None,
    *,
    token_count: int | None = None,
    session_index: int | None = None,
    trajectory_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "source": candidate["source"],
        "task_id": candidate["task_id"],
        "role": role,
        "session_index": session_index,
        "accepted": not reasons,
        "task_score": score,
        "token_count": token_count,
        "reasons": sorted(set(reasons)),
        "trajectory_sha256": trajectory_sha256,
    }


def _solver_trace(
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    workspace: Path,
    score: float,
    seen: set[str],
    count_tokens: TokenCounter,
    cutoff_len: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    messages_path = workspace / "artifacts" / "messages.json"
    raw_messages = read_json(messages_path) if messages_path.is_file() else None
    messages = None
    feedback_removed = 0
    if isinstance(raw_messages, list):
        messages, feedback_removed = strip_feedback_attempts(raw_messages)
    audit = audit_solver_episode(
        {"status": "completed", "solver_messages": messages},
        token_counter=count_tokens,
        cutoff_len=cutoff_len,
    )
    reasons = list(audit.reasons)
    trace_hash = None
    record = None
    if not reasons and isinstance(messages, list):
        trace_hash = trajectory_hash(messages)
        if trace_hash in seen:
            reasons.append("duplicate_messages")
        else:
            seen.add(trace_hash)
            record = sft_record(
                messages=messages,
                role="solver",
                candidate=candidate,
                result=result,
                score=score,
                token_count=int(audit.token_count or 0),
                trajectory_hash=trace_hash,
                feedback_pairs_removed=feedback_removed,
            )
    return record, _manifest_row(
        candidate,
        "solver",
        reasons,
        score,
        token_count=audit.token_count,
        trajectory_sha256=trace_hash,
    )


def _perception_traces(
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    workspace: Path,
    score: float,
    seen: set[str],
    count_tokens: TokenCounter,
    cutoff_len: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    sessions_path = workspace / "artifacts" / "perception_sessions.json"
    sessions = read_json(sessions_path) if sessions_path.is_file() else []
    if not isinstance(sessions, list):
        return (
            [],
            [
                _manifest_row(
                    candidate,
                    "perception",
                    ["invalid_perception_sessions"],
                    score,
                )
            ],
            False,
        )

    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            manifest.append(
                _manifest_row(
                    candidate,
                    "perception",
                    ["invalid_session"],
                    score,
                    session_index=session_index,
                )
            )
            continue
        raw_messages = session.get("messages")
        messages = None
        feedback_removed = 0
        if isinstance(raw_messages, list):
            messages, feedback_removed = strip_feedback_attempts(raw_messages)
        audit = audit_perception_record(
            {
                "messages": messages,
                "metadata": {"session_status": session.get("status")},
            },
            token_counter=count_tokens,
            cutoff_len=cutoff_len,
        )
        reasons = list(audit.reasons)
        trace_hash = None
        if not reasons and isinstance(messages, list):
            trace_hash = trajectory_hash(messages)
            if trace_hash in seen:
                reasons.append("duplicate_messages")
            else:
                seen.add(trace_hash)
                records.append(
                    sft_record(
                        messages=messages,
                        role="perception",
                        candidate=candidate,
                        result=result,
                        score=score,
                        token_count=int(audit.token_count or 0),
                        trajectory_hash=trace_hash,
                        feedback_pairs_removed=feedback_removed,
                        session=session,
                        session_index=session_index,
                    )
                )
        manifest.append(
            _manifest_row(
                candidate,
                "perception",
                reasons,
                score,
                token_count=audit.token_count,
                session_index=session_index,
                trajectory_sha256=trace_hash,
            )
        )
    return records, manifest, bool(sessions)


def export_perception_cold_start_sft(
    *,
    run_root: Path,
    candidates_path: Path,
    output_dir: Path,
    token_counter: TokenCounter,
    tokenizer_name: str,
    cutoff_len: int = 24_576,
    seed: int = 42,
    smoke_samples: int = 8,
    additional_run_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    if cutoff_len <= 0:
        raise ValueError("cutoff_len must be positive")
    candidates = read_jsonl(candidates_path.expanduser().resolve())
    identities = [(str(row.get("source")), str(row.get("task_id"))) for row in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Candidate source/task identities must be unique")
    if any(source not in SOURCES or not task_id for source, task_id in identities):
        raise ValueError("Candidate manifest contains an unsupported source or empty task ID")

    run_root = run_root.expanduser().resolve()
    run_roots = [
        run_root,
        *(path.expanduser().resolve() for path in additional_run_roots),
    ]
    if len(run_roots) != len(set(run_roots)):
        raise ValueError("Run roots must be unique")
    output_dir = output_dir.expanduser().resolve()
    results = result_index(run_roots)
    seen: dict[str, set[str]] = {"solver": set(), "perception": set()}
    solver_records: list[dict[str, Any]] = []
    perception_records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    successful_tasks: set[tuple[str, str]] = set()
    tasks_with_perception: set[tuple[str, str]] = set()

    for candidate in candidates:
        source = str(candidate["source"])
        task_id = str(candidate["task_id"])
        key = (source, task_id)
        result_entry = results.get(key)
        if result_entry is None:
            manifest.append(_manifest_row(candidate, "solver", ["missing_result"], None))
            continue
        _, result = result_entry
        score, gate_error = task_gate(source, result)
        if gate_error is not None:
            manifest.append(_manifest_row(candidate, "solver", [gate_error], score))
            continue
        assert score is not None
        successful_tasks.add(key)

        workspace, _, runtime_error = runtime_artifacts(result)
        if runtime_error is not None or workspace is None:
            manifest.append(
                _manifest_row(
                    candidate,
                    "solver",
                    [runtime_error or "missing_workspace"],
                    score,
                )
            )
            continue

        solver_record, solver_manifest = _solver_trace(
            candidate,
            result,
            workspace,
            score,
            seen["solver"],
            token_counter,
            cutoff_len,
        )
        manifest.append(solver_manifest)
        if solver_record is not None:
            solver_records.append(solver_record)

        records, rows, triggered = _perception_traces(
            candidate,
            result,
            workspace,
            score,
            seen["perception"],
            token_counter,
            cutoff_len,
        )
        perception_records.extend(records)
        manifest.extend(rows)
        if triggered:
            tasks_with_perception.add(key)

    if smoke_samples <= 0:
        raise ValueError("smoke_samples must be positive")
    joint_records, paths = write_sft_dataset(
        output_dir,
        solver_records=solver_records,
        perception_records=perception_records,
        manifest_rows=manifest,
        manifest_name="cleaning_manifest.jsonl",
        seed=seed,
        smoke_samples=smoke_samples,
    )

    rejection_reasons = Counter(
        reason for row in manifest if not row["accepted"] for reason in row["reasons"]
    )
    accepted_by_role_source = {
        role: dict(
            sorted(Counter(record["metadata"]["source"] for record in records_for_role).items())
        )
        for role, records_for_role in (
            ("solver", solver_records),
            ("perception", perception_records),
        )
    }
    summary = {
        "schema_version": 1,
        "dataset_name": DATASET_NAME,
        "seed": seed,
        "cutoff_len": cutoff_len,
        "tokenizer": tokenizer_name,
        "inputs": {
            "run_root": str(run_root),
            "run_roots": [str(path) for path in run_roots],
            "candidates": str(candidates_path.expanduser().resolve()),
            "candidate_tasks": len(candidates),
            "observed_results": len(results),
        },
        "policy": {
            "dare_bench": "status=completed and score>=0.5",
            "data_agent_rl": "status=completed and exact reward=1",
            "datamind_sql": "status=completed and exact reward=1",
            "solver": (
                "completed strict ReAct trajectory with terminal Answer after "
                "dropping rejected assistant/Feedback pairs"
            ),
            "perception": "completed strict read-only Perception trajectory with terminal Report",
            "common": "thinking=false, perception=true, unique messages, within cutoff",
        },
        "tasks": {
            "programmatically_successful": len(successful_tasks),
            "programmatically_successful_by_source": dict(
                sorted(Counter(source for source, _ in successful_tasks).items())
            ),
            "triggered_perception": len(tasks_with_perception),
            "triggered_perception_by_source": dict(
                sorted(Counter(source for source, _ in tasks_with_perception).items())
            ),
            "perception_trigger_rate_among_successful": (
                len(tasks_with_perception) / len(successful_tasks) if successful_tasks else 0.0
            ),
        },
        "accepted": {
            "solver": len(solver_records),
            "perception": len(perception_records),
            "total": len(joint_records),
            "by_role_source": accepted_by_role_source,
        },
        "rejected": {
            "manifest_rows": sum(not row["accepted"] for row in manifest),
            "reason_counts": dict(sorted(rejection_reasons.items())),
        },
        "validation": {
            "unique_trajectories": len(joint_records)
            == len({record["metadata"]["trajectory_sha256"] for record in joint_records}),
            "all_within_cutoff": all(
                int(record["metadata"]["chat_template_tokens"]) <= cutoff_len
                for record in joint_records
            ),
            "all_task_rewards_pass": all(
                float(record["metadata"]["task_score"])
                >= float(record["metadata"]["task_success_threshold"])
                for record in joint_records
            ),
            "all_perception_completed": all(
                record["metadata"].get("session_status") == "completed"
                for record in perception_records
            ),
            "roles": sorted({record["metadata"]["role"] for record in joint_records}),
            "max_tokens": max(
                int(record["metadata"]["chat_template_tokens"]) for record in joint_records
            ),
        },
        "trainable": {
            "format": "LLaMA-Factory ShareGPT messages",
            "dataset": DATASET_NAME,
            "records": len(joint_records),
            "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
            "smoke_records": smoke_samples,
        },
        "outputs": {
            "solver": str(paths["solver"]),
            "perception": str(paths["perception"]),
            "joint": str(paths["joint"]),
            "joint_jsonl": str(paths["joint_jsonl"]),
            "manifest": str(paths["manifest"]),
            "llamafactory_dir": str(paths["train"].parent),
            "train": str(paths["train"]),
            "dataset_info": str(paths["dataset_info"]),
            "smoke": str(paths["smoke"]),
            "summary": str(paths["summary"]),
        },
    }
    write_json(paths["summary"], summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--additional-run-root", type=Path, action="append", default=[])
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=24_576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-samples", type=int, default=8)
    args = parser.parse_args(argv)
    summary = export_perception_cold_start_sft(
        run_root=args.run_root,
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        token_counter=token_counter(args.tokenizer),
        tokenizer_name=str(args.tokenizer.expanduser().resolve()),
        cutoff_len=args.cutoff_len,
        seed=args.seed,
        smoke_samples=args.smoke_samples,
        additional_run_roots=args.additional_run_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
