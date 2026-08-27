"""Validate Solver and Perception trajectories before SFT export."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dslighting.workflows.search.react.perception_protocol import (
    normalize_perception_reply,
    parse_perception_reply,
)
from dslighting.workflows.search.react.protocol import (
    extract_action_block,
    extract_strict_python_from_action,
    normalize_react_reply,
    parse_react_reply,
    validate_turn_structure,
)

TokenCounter = Callable[[list[dict[str, Any]]], int]

_SOLVER_REPLY_PATTERN = re.compile(
    r"\A<Think>.*?</Think>\s*"
    r"(?:<Action>.*</Action>|<Explore>.*</Explore>|<Answer>.*</Answer>)\Z",
    re.DOTALL,
)
_MUTATING_METHODS = {
    "save",
    "savefig",
    "savetxt",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}


@dataclass(frozen=True)
class AuditResult:
    reasons: tuple[str, ...]
    token_count: int | None

    @property
    def accepted(self) -> bool:
        return not self.reasons


def _valid_role_sequence(messages: list[dict[str, Any]]) -> bool:
    if len(messages) < 3:
        return False
    if [message.get("role") for message in messages[:2]] != ["system", "user"]:
        return False
    return all(
        message.get("role") == ("assistant" if index % 2 == 0 else "user")
        for index, message in enumerate(messages[2:], start=2)
    )


def _contains_protocol_feedback(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "user"
        and (
            "<Feedback>" in str(message.get("content") or "")
            or "protocol error" in str(message.get("content") or "").casefold()
        )
        for message in messages
    )


def _python_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _python_may_write(code: str) -> bool:
    """Conservatively reject obvious filesystem writes in Perception actions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in _MUTATING_METHODS:
            return True
        if name == "open":
            mode: str | None = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = str(keyword.value.value)
            if mode is not None and any(flag in mode for flag in "wax+"):
                return True
    return False


def _count_tokens(
    messages: list[dict[str, Any]],
    *,
    token_counter: TokenCounter,
    reasons: set[str],
) -> int | None:
    try:
        count = int(token_counter(messages))
    except (KeyError, RuntimeError, TypeError, ValueError):
        reasons.add("tokenization_error")
        return None
    if count <= 0:
        reasons.add("tokenization_error")
        return None
    return count


def audit_solver_episode(
    episode: Mapping[str, Any],
    *,
    token_counter: TokenCounter,
    cutoff_len: int,
) -> AuditResult:
    reasons: set[str] = set()
    messages = episode.get("solver_messages")
    if not isinstance(messages, list):
        return AuditResult(("missing_solver_messages",), None)
    if episode.get("status") != "completed":
        reasons.add("not_completed")
    if not _valid_role_sequence(messages):
        reasons.add("invalid_role_sequence")
    if _contains_protocol_feedback(messages):
        reasons.add("contains_protocol_feedback")

    assistant_replies = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if not assistant_replies:
        reasons.add("missing_assistant_reply")
    for message in assistant_replies:
        content = message.get("content")
        if not isinstance(content, str):
            reasons.add("non_string_assistant_reply")
            continue
        normalized = normalize_react_reply(content)
        if normalized.repaired:
            reasons.add("silently_repaired")
        valid, _ = validate_turn_structure(content, allow_explore=True)
        if not valid or not _SOLVER_REPLY_PATTERN.fullmatch(content.strip()):
            reasons.add("invalid_protocol")
        parsed = parse_react_reply(content, allow_explore=True)
        action = extract_action_block(content)
        if action is not None and extract_strict_python_from_action(action) is None:
            reasons.add("invalid_protocol")
        if parsed.next_user_message is not None:
            reasons.add("parser_feedback")
        if parsed.action_code is not None and not _python_is_valid(parsed.action_code):
            reasons.add("invalid_python")

    if assistant_replies:
        final_content = assistant_replies[-1].get("content")
        final = parse_react_reply(
            final_content if isinstance(final_content, str) else "",
            allow_explore=True,
        )
        if messages[-1].get("role") != "assistant" or final.final_answer is None:
            reasons.add("missing_terminal_answer")

    token_count = _count_tokens(messages, token_counter=token_counter, reasons=reasons)
    if token_count is not None and token_count > cutoff_len:
        reasons.add("over_cutoff")
    return AuditResult(tuple(sorted(reasons)), token_count)


def audit_perception_record(
    record: Mapping[str, Any],
    *,
    token_counter: TokenCounter,
    cutoff_len: int,
) -> AuditResult:
    reasons: set[str] = set()
    messages = record.get("messages")
    metadata = record.get("metadata")
    if not isinstance(messages, list):
        return AuditResult(("missing_messages",), None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    if metadata.get("session_status") != "completed":
        reasons.add("not_completed")
    if not _valid_role_sequence(messages):
        reasons.add("invalid_role_sequence")
    if _contains_protocol_feedback(messages):
        reasons.add("contains_protocol_feedback")

    assistant_replies = [
        message for message in messages if message.get("role") == "assistant"
    ]
    parsed_replies = []
    if not assistant_replies:
        reasons.add("missing_assistant_reply")
    for message in assistant_replies:
        content = message.get("content")
        if not isinstance(content, str):
            reasons.add("non_string_assistant_reply")
            continue
        normalized = normalize_perception_reply(content)
        parsed = parse_perception_reply(content)
        parsed_replies.append(parsed)
        if normalized.repaired:
            reasons.add("silently_repaired")
        if parsed.next_user_message is not None:
            reasons.add("invalid_protocol")
        if parsed.action_code is not None:
            if not _python_is_valid(parsed.action_code):
                reasons.add("invalid_python")
            elif _python_may_write(parsed.action_code):
                reasons.add("file_write")

    if (
        not parsed_replies
        or parsed_replies[-1].report is None
        or messages[-1].get("role") != "assistant"
    ):
        reasons.add("missing_terminal_report")
    if any(reply.report is not None for reply in parsed_replies[:-1]):
        reasons.add("report_before_terminal_turn")

    token_count = _count_tokens(messages, token_counter=token_counter, reasons=reasons)
    if token_count is not None and token_count > cutoff_len:
        reasons.add("over_cutoff")
    return AuditResult(tuple(sorted(reasons)), token_count)
