"""Export successful, isolated Perception sessions as ShareGPT SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dslighting.workflows.search.react.perception_protocol import parse_perception_reply


def _episode_paths(root: Path) -> list[Path]:
    return sorted(Path(root).expanduser().resolve().glob("*/episode.json"))


def _valid_session(session: Mapping[str, Any]) -> bool:
    messages = session.get("messages")
    if session.get("agent_role") != "perception" or session.get("status") != "completed":
        return False
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return False
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        return False
    for message in assistant_messages:
        content = message.get("content")
        if not isinstance(content, str):
            return False
        parsed = parse_perception_reply(content)
        if parsed.action_code is None and parsed.report is None:
            return False
    return parse_perception_reply(assistant_messages[-1]["content"]).report is not None


def export_sft(
    episode_root: Path,
    output_path: Path,
    *,
    require_judge: bool = True,
    min_judge_score: float = 1.0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {
        "episodes_seen": 0,
        "episodes_accepted": 0,
        "sessions_seen": 0,
        "sessions_exported": 0,
        "duplicate_sessions": 0,
        "invalid_sessions": 0,
        "rejected_by_judge": 0,
    }
    for path in _episode_paths(episode_root):
        episode = json.loads(path.read_text(encoding="utf-8"))
        stats["episodes_seen"] += 1
        judge = episode.get("judge")
        score = judge.get("score") if isinstance(judge, Mapping) else None
        if require_judge and (score is None or float(score) < min_judge_score):
            stats["rejected_by_judge"] += 1
            continue
        stats["episodes_accepted"] += 1
        sessions = episode.get("perception_sessions") or []
        for session in sessions:
            stats["sessions_seen"] += 1
            if not isinstance(session, Mapping) or not _valid_session(session):
                stats["invalid_sessions"] += 1
                continue
            messages = session["messages"]
            digest = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                stats["duplicate_sessions"] += 1
                continue
            seen.add(digest)
            records.append(
                {
                    "messages": messages,
                    "metadata": {
                        "episode_id": episode.get("episode_id"),
                        "task_id": episode.get("task_id"),
                        "source_task_id": episode.get("source_task_id"),
                        "agent_role": "perception",
                        "segment_id": session.get("segment_id"),
                        "parent_solver_step": session.get("parent_solver_step"),
                        "final_reward": score,
                    },
                }
            )
            stats["sessions_exported"] += 1
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stats["output"] = str(output_path)
    stats["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unjudged", action="store_true")
    parser.add_argument("--min-judge-score", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    stats = export_sft(
        args.episode_root,
        args.output,
        require_judge=not args.allow_unjudged,
        min_judge_score=args.min_judge_score,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
