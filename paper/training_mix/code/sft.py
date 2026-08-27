"""Shared record rules for the Training Mix SFT exports."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCES = ("dare_bench", "data_agent_rl", "datamind_sql")
SOURCE_SUCCESS_THRESHOLDS = {
    "dare_bench": 0.5,
    "data_agent_rl": 1.0,
    "datamind_sql": 1.0,
}
DATASET_NAME = "training_mix_perception_joint_v1"

TokenCounter = Callable[[list[dict[str, Any]]], int]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def trajectory_hash(messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def strip_feedback_attempts(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drop rejected assistant reply + runtime Feedback pairs, keeping the retry."""

    cleaned: list[dict[str, Any]] = []
    removed = 0
    index = 0
    while index < len(messages):
        current = messages[index]
        if (
            index + 1 < len(messages)
            and current.get("role") == "assistant"
            and messages[index + 1].get("role") == "user"
            and "<Feedback>" in str(messages[index + 1].get("content") or "")
        ):
            removed += 1
            index += 2
            continue
        cleaned.append(dict(current))
        index += 1
    return cleaned, removed


def result_index(
    run_roots: Sequence[Path],
) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    index: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for run_root in run_roots:
        repeat_root = run_root / "repetitions" / "repeat_00"
        for source in SOURCES:
            for path in sorted((repeat_root / source).glob("*/result.json")):
                payload = read_json(path)
                if not isinstance(payload, dict):
                    raise ValueError(f"Result must be an object: {path}")
                task_id = str(payload.get("task_id") or "").strip()
                key = (source, task_id)
                if not task_id or key in index:
                    raise ValueError(f"Missing or duplicate result identity: {path}")
                index[key] = (path, payload)
    return index


def task_score(source: str, result: Mapping[str, Any]) -> float | None:
    field = "score" if source == "dare_bench" else "reward"
    value = result.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def task_gate(source: str, result: Mapping[str, Any]) -> tuple[float | None, str | None]:
    score = task_score(source, result)
    if result.get("status") != "completed":
        return score, f"status_{result.get('status', 'unknown')}"
    if score is None:
        return None, "missing_score"
    if score < SOURCE_SUCCESS_THRESHOLDS[source]:
        return score, "task_reward_below_threshold"
    return score, None


def runtime_artifacts(
    result: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    dslighting = result.get("dslighting")
    if not isinstance(dslighting, Mapping):
        return None, None, "missing_dslighting_metadata"
    workspace = Path(str(dslighting.get("workspace_dir") or ""))
    if not workspace.is_dir():
        return None, None, "missing_workspace"
    metadata_path = workspace / "artifacts" / "telemetry" / "run_metadata.json"
    if not metadata_path.is_file():
        return workspace, None, "missing_run_metadata"
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        return workspace, None, "invalid_run_metadata"
    snapshot = metadata.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        return workspace, metadata, "missing_config_snapshot"
    llm = snapshot.get("llm")
    runtime = snapshot.get("agent_runtime")
    if not isinstance(llm, Mapping) or llm.get("thinking") is not False:
        return workspace, metadata, "thinking_not_disabled"
    if not isinstance(runtime, Mapping) or runtime.get("perception_enabled") is not True:
        return workspace, metadata, "perception_not_enabled"
    return workspace, metadata, None


def token_counter(tokenizer_path: Path) -> TokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path.expanduser().resolve(),
        local_files_only=True,
    )

    def count(messages: list[dict[str, Any]]) -> int:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)

    return count


def sft_record(
    *,
    messages: list[dict[str, Any]],
    role: str,
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    score: float,
    token_count: int,
    trajectory_hash: str,
    feedback_pairs_removed: int,
    session: Mapping[str, Any] | None = None,
    session_index: int | None = None,
) -> dict[str, Any]:
    dslighting = result.get("dslighting")
    dslighting = dslighting if isinstance(dslighting, Mapping) else {}
    usage = dslighting.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    metadata = {
        "role": role,
        "source": candidate["source"],
        "task_id": candidate["task_id"],
        "taxonomy_primary": candidate.get("taxonomy_primary"),
        "task_family": candidate.get("task_family"),
        "runtime_heavy": bool(candidate.get("runtime_heavy", False)),
        "task_score": score,
        "task_success_threshold": SOURCE_SUCCESS_THRESHOLDS[str(candidate["source"])],
        "teacher_model": dslighting.get("model"),
        "thinking": False,
        "perception_enabled": True,
        "chat_template_tokens": token_count,
        "trajectory_sha256": trajectory_hash,
        "feedback_pairs_removed": feedback_pairs_removed,
        "episode_call_count": usage.get("call_count"),
        "episode_total_tokens": usage.get("total_tokens"),
    }
    if session is not None:
        metadata.update(
            session_index=session_index,
            segment_id=session.get("segment_id"),
            parent_solver_step=session.get("parent_solver_step"),
            session_status=session.get("status"),
            exploration_request=session.get("request"),
        )
    return {"messages": messages, "metadata": metadata}


def write_sft_dataset(
    output_dir: Path,
    *,
    solver_records: Sequence[Mapping[str, Any]],
    perception_records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_name: str,
    seed: int,
    smoke_samples: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Write the canonical clean and LLaMA-Factory views once."""

    records = [dict(row) for row in (*solver_records, *perception_records)]
    if not records:
        raise ValueError("No trainable Solver or Perception records were accepted")
    random.Random(seed).shuffle(records)
    if smoke_samples < 0 or smoke_samples > len(records):
        raise ValueError("smoke_samples must be within the accepted dataset size")
    smoke = sorted(
        records,
        key=lambda row: int(row["metadata"]["chat_template_tokens"]),
        reverse=True,
    )[:smoke_samples]

    clean_dir = output_dir / "clean"
    llamafactory_dir = output_dir / "llamafactory"
    paths = {
        "solver": clean_dir / "solver_sft_clean.json",
        "perception": clean_dir / "perception_sft_clean.json",
        "joint": clean_dir / "joint_sft_train.json",
        "joint_jsonl": clean_dir / "joint_sft_train.jsonl",
        "manifest": clean_dir / manifest_name,
        "train": llamafactory_dir / "train.json",
        "smoke": llamafactory_dir / "smoke_longest.json",
        "dataset_info": llamafactory_dir / "dataset_info.json",
        "summary": output_dir / "export_summary.json",
    }
    write_json(paths["solver"], solver_records)
    write_json(paths["perception"], perception_records)
    write_json(paths["joint"], records)
    write_jsonl(paths["joint_jsonl"], records)
    write_jsonl(paths["manifest"], manifest_rows)
    write_json(paths["train"], records)
    if smoke_samples:
        write_json(paths["smoke"], smoke)

    tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    }
    datasets = {
        DATASET_NAME: {
            "file_name": paths["train"].name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": tags,
        }
    }
    if smoke_samples:
        datasets[f"{DATASET_NAME}_smoke"] = {
            "file_name": paths["smoke"].name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": tags,
        }
    write_json(paths["dataset_info"], datasets)
    return records, paths


__all__ = [
    "DATASET_NAME",
    "SOURCES",
    "SOURCE_SUCCESS_THRESHOLDS",
    "TokenCounter",
    "read_json",
    "read_jsonl",
    "result_index",
    "runtime_artifacts",
    "sft_record",
    "strip_feedback_attempts",
    "task_gate",
    "task_score",
    "token_counter",
    "trajectory_hash",
    "write_json",
    "write_jsonl",
    "write_sft_dataset",
]
