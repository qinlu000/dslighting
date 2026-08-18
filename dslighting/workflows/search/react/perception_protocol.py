"""Protocol helpers for the lightweight Perception Agent loop."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dslighting.workflows.search.react.protocol import wrap_feedback

_ACTION_PATTERN = re.compile(
    r"\s*<Action>(?P<body>.*?)</Action>\s*",
    re.DOTALL | re.IGNORECASE,
)
_REPORT_PATTERN = re.compile(
    r"\s*<Report>(?P<body>.*?)</Report>\s*",
    re.DOTALL | re.IGNORECASE,
)
_PYTHON_BLOCK_PATTERN = re.compile(
    r"\s*```python\s*(?P<code>.*?)\s*```\s*",
    re.DOTALL | re.IGNORECASE,
)
_THINK_PREFIX_PATTERN = re.compile(
    r"\s*<Think>.*?</Think>\s*(?P<response>.*)\s*",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedPerceptionReply:
    """Normalized Perception reply plus low-risk repair metadata."""

    raw_content: str
    normalized_content: str
    repaired: bool = False
    repair_reason: str | None = None


@dataclass(frozen=True)
class PerceptionTurnResult:
    """Parsed outcome of one Perception Agent reply."""

    action_code: str | None = None
    report: str | None = None
    next_user_message: str | None = None


def normalize_perception_reply(content: str) -> NormalizedPerceptionReply:
    """Repair one unambiguous Action or Report block missing one boundary tag."""
    raw_content = content if isinstance(content, str) else str(content)
    present_tags = [tag for tag in ("Action", "Report") if _has_tag(raw_content, tag)]
    if len(present_tags) != 1:
        return NormalizedPerceptionReply(raw_content, raw_content)

    tag = present_tags[0]
    open_count = _tag_count(raw_content, tag, closing=False)
    close_count = _tag_count(raw_content, tag, closing=True)
    if open_count == 1 and close_count == 0:
        repaired = raw_content.rstrip() + f"\n</{tag}>"
        reason = f"added missing </{tag}> closing tag"
    elif open_count == 0 and close_count == 1:
        close_match = re.search(rf"</{re.escape(tag)}>", raw_content, flags=re.IGNORECASE)
        if close_match is None:
            return NormalizedPerceptionReply(raw_content, raw_content)
        prefix = raw_content[: close_match.start()]
        think_closes = list(re.finditer(r"</Think>", prefix, flags=re.IGNORECASE))
        if think_closes:
            opening_index = think_closes[-1].end()
        elif tag == "Action":
            payload_match = re.search(r"```python\b", prefix, flags=re.IGNORECASE)
            if payload_match is None:
                return NormalizedPerceptionReply(raw_content, raw_content)
            opening_index = payload_match.start()
        else:
            opening_index = len(prefix) - len(prefix.lstrip())
        repaired = (
            raw_content[:opening_index]
            + f"<{tag}>"
            + raw_content[opening_index:]
        )
        reason = f"added missing <{tag}> opening tag"
    else:
        return NormalizedPerceptionReply(raw_content, raw_content)
    if not _has_valid_payload(repaired, tag):
        return NormalizedPerceptionReply(raw_content, raw_content)
    return NormalizedPerceptionReply(
        raw_content=raw_content,
        normalized_content=repaired,
        repaired=True,
        repair_reason=reason,
    )


def parse_perception_reply(content: str) -> PerceptionTurnResult:
    """Parse one reply as either a Python action or a plain-text report."""
    reply = normalize_perception_reply(content).normalized_content
    response = _strip_optional_think_prefix(reply)
    if response is None:
        return _protocol_error(
            "The <Think>...</Think> block is malformed or is not followed by a response block."
        )
    action_open = _tag_count(reply, "Action", closing=False)
    action_close = _tag_count(reply, "Action", closing=True)
    report_open = _tag_count(reply, "Report", closing=False)
    report_close = _tag_count(reply, "Report", closing=True)
    has_action = action_open > 0 or action_close > 0
    has_report = report_open > 0 or report_close > 0

    if has_action == has_report:
        return _protocol_error(
            "Reply must contain exactly one of <Action>...</Action> or "
            "<Report>...</Report>."
        )

    if has_action:
        if action_open != 1 or action_close != 1:
            return _protocol_error(
                "Reply must contain exactly one <Action>...</Action> block."
            )
        match = _ACTION_PATTERN.fullmatch(response)
        if match is None:
            return _protocol_error(
                "Reply must contain only <Action>...</Action> with no extra text."
            )
        code_match = _PYTHON_BLOCK_PATTERN.fullmatch(match.group("body"))
        if code_match is None or not code_match.group("code").strip():
            return _protocol_error(
                "<Action> must contain exactly one non-empty fenced "
                "```python``` block."
            )
        return PerceptionTurnResult(action_code=code_match.group("code").strip())

    if report_open != 1 or report_close != 1:
        return _protocol_error(
            "Reply must contain exactly one <Report>...</Report> block."
        )
    match = _REPORT_PATTERN.fullmatch(response)
    if match is None:
        return _protocol_error(
            "Reply must contain only <Report>...</Report> with no extra text."
        )
    report = match.group("body").strip()
    if not report:
        return _protocol_error("<Report> cannot be empty.")
    if "```" in report:
        return _protocol_error("<Report> must be plain text without code blocks.")
    return PerceptionTurnResult(report=report)


def _has_valid_payload(reply: str, tag: str) -> bool:
    response = _strip_optional_think_prefix(reply)
    if response is None:
        return False
    if tag == "Action":
        match = _ACTION_PATTERN.fullmatch(response)
        if match is None:
            return False
        code_match = _PYTHON_BLOCK_PATTERN.fullmatch(match.group("body"))
        return bool(code_match and code_match.group("code").strip())

    match = _REPORT_PATTERN.fullmatch(response)
    if match is None:
        return False
    report = match.group("body").strip()
    return bool(report and "```" not in report)


def _strip_optional_think_prefix(reply: str) -> str | None:
    """Return the response block while tolerating a missing Think envelope."""
    think_open = _tag_count(reply, "Think", closing=False)
    think_close = _tag_count(reply, "Think", closing=True)
    if think_open == 0 and think_close == 0:
        return reply.strip()
    if think_open != 1 or think_close != 1:
        return None
    match = _THINK_PREFIX_PATTERN.fullmatch(reply)
    if match is None:
        return None
    return match.group("response").strip()


def _tag_count(content: str, tag: str, *, closing: bool) -> int:
    slash = "/" if closing else ""
    return len(
        re.findall(
            rf"<{slash}{re.escape(tag)}>",
            content,
            flags=re.IGNORECASE,
        )
    )


def _has_tag(content: str, tag: str) -> bool:
    return bool(
        _tag_count(content, tag, closing=False)
        or _tag_count(content, tag, closing=True)
    )


def _protocol_error(reason: str) -> PerceptionTurnResult:
    feedback = (
        f"Perception protocol error: {reason}\n"
        "Reply with exactly two blocks and no other text:\n"
        "<Think>reason about the next step</Think>\n"
        "<Action>```python\n# read-only inspection code\n```</Action>\n"
        "or:\n"
        "<Think>reason about the findings</Think>\n"
        "<Report>concise answer and relevant data findings</Report>"
    )
    return PerceptionTurnResult(next_user_message=wrap_feedback(feedback))


__all__ = [
    "NormalizedPerceptionReply",
    "PerceptionTurnResult",
    "normalize_perception_reply",
    "parse_perception_reply",
]
