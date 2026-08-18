"""Protocol helpers for the ReAct workflow."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional

from dslighting.utils.typing import ExecutionResult


@dataclass(frozen=True)
class NormalizedReActReply:
    """Normalized assistant reply plus repair metadata."""

    raw_content: str
    normalized_content: str
    repaired: bool = False
    repair_reason: str | None = None


@dataclass(frozen=True)
class ReActTurnResult:
    """Parsed outcome of one assistant reply in the ReAct protocol."""

    next_user_message: str | None = None
    final_answer: str | None = None
    execution_succeeded: bool = False
    action_code: str | None = None
    explore_request: str | None = None


_STRICT_PYTHON_BLOCK_PATTERN = re.compile(
    r"\s*```python\s*(?P<code>.*?)\s*```\s*",
    re.DOTALL | re.IGNORECASE,
)


def parse_react_reply(
    content: str,
    *,
    allow_explore: bool = True,
) -> ReActTurnResult:
    """Parse one assistant reply into either code, a final answer, or protocol feedback."""
    normalized = normalize_react_reply(content)
    is_valid, reason = validate_turn_structure(
        normalized.normalized_content,
        allow_explore=allow_explore,
    )
    if not is_valid:
        return ReActTurnResult(
            next_user_message=wrap_feedback(
                build_protocol_error_feedback(reason, allow_explore=allow_explore)
            )
        )

    action = extract_action_block(normalized.normalized_content)
    if action is not None:
        code = extract_strict_python_from_action(action)
        if code is not None:
            return ReActTurnResult(action_code=code)

        final_answer = extract_final_answer_from_action(action)
        if final_answer is None:
            return ReActTurnResult(
                next_user_message=wrap_feedback(
                    build_protocol_error_feedback(
                        "Code actions must contain exactly one fenced ```python``` block "
                        "with no extra text before or after it.",
                        allow_explore=allow_explore,
                    )
                )
            )
        return ReActTurnResult(final_answer=final_answer)

    explore_request = extract_explore_block(normalized.normalized_content)
    if explore_request is not None:
        return ReActTurnResult(explore_request=explore_request)

    answer = extract_answer_block(normalized.normalized_content)
    if answer is None:
        return ReActTurnResult(
            next_user_message=wrap_feedback(
                build_protocol_error_feedback(
                    "Missing <Action>...</Action>, <Explore>...</Explore>, "
                    "or <Answer>...</Answer> block.",
                    allow_explore=allow_explore,
                )
            )
        )
    return ReActTurnResult(final_answer=answer)


def build_execution_message(
    exec_result: ExecutionResult,
    *,
    obs_max_tokens: int,
    obs_head_tokens: int,
    obs_tail_tokens: int,
    critical_footer: str | None = None,
    escape_output: bool = False,
) -> str:
    """Render execution output back into the ReAct conversation."""
    raw_output = format_observation(exec_result)
    if escape_output:
        raw_output = html.escape(raw_output, quote=False)
    execution_output = truncate_observation(
        raw_output,
        obs_max_tokens=obs_max_tokens,
        obs_head_tokens=obs_head_tokens,
        obs_tail_tokens=obs_tail_tokens,
    )
    if critical_footer:
        observation = (
            "<ExecutionOutput>\n"
            f"{execution_output}\n"
            "</ExecutionOutput>\n"
            f"{critical_footer.strip()}"
        )
    else:
        observation = execution_output
    return wrap_observation(observation)


def build_perception_message(
    perception_result: str,
    *,
    obs_max_tokens: int,
    obs_head_tokens: int,
    obs_tail_tokens: int,
) -> str:
    """Render a bounded Perception result back into the Solving Agent context."""
    perception = truncate_observation(
        html.escape(perception_result, quote=False),
        obs_max_tokens=obs_max_tokens,
        obs_head_tokens=obs_head_tokens,
        obs_tail_tokens=obs_tail_tokens,
    )
    return wrap_observation(f"<PerceptionResult>\n{perception}\n</PerceptionResult>")


def normalize_react_reply(content: str) -> NormalizedReActReply:
    """Repair only low-risk protocol shell issues for assistant replies."""
    raw_content = content if isinstance(content, str) else str(content)

    if _has_tag(raw_content, "Final Answer"):
        return NormalizedReActReply(
            raw_content=raw_content,
            normalized_content=raw_content,
        )

    shell_repair = _repair_response_shell(raw_content)
    if shell_repair is not None:
        repaired_content, repair_reason = shell_repair
        return NormalizedReActReply(
            raw_content=raw_content,
            normalized_content=repaired_content,
            repaired=True,
            repair_reason=repair_reason,
        )

    return NormalizedReActReply(
        raw_content=raw_content,
        normalized_content=raw_content,
    )


def extract_tag_block(content: str, tag: str) -> Optional[str]:
    pattern = re.compile(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(content)
    if not match:
        return None
    return match.group(1).strip()


def extract_think_block(content: str) -> Optional[str]:
    return extract_tag_block(content, "Think")


def extract_action_block(content: str) -> Optional[str]:
    return extract_tag_block(content, "Action")


def extract_explore_block(content: str) -> Optional[str]:
    return extract_tag_block(content, "Explore")


def extract_answer_block(content: str) -> Optional[str]:
    return extract_tag_block(content, "Answer")


def extract_strict_python_from_action(action: str) -> Optional[str]:
    match = _STRICT_PYTHON_BLOCK_PATTERN.fullmatch(action)
    if not match:
        return None
    return match.group("code").strip()


def extract_final_answer_from_action(action: str) -> Optional[str]:
    stripped = action.strip()
    if not stripped or "```" in stripped:
        return None
    return stripped


def validate_turn_structure(
    content: str,
    *,
    allow_explore: bool = True,
) -> tuple[bool, Optional[str]]:
    """Validate one response action while leaving the reasoning envelope free-form."""
    # Response delimiters are reserved tokens. Count both sides before extracting
    # a block so regex backtracking cannot absorb duplicate or nested actions.
    if _has_tag(content, "Final Answer"):
        return (
            False,
            "<Final Answer>...</Final Answer> is not supported. Use <Answer>...</Answer> instead.",
        )

    action_open_count = _tag_count(content, "Action", closing=False)
    action_close_count = _tag_count(content, "Action", closing=True)
    explore_open_count = _tag_count(content, "Explore", closing=False)
    explore_close_count = _tag_count(content, "Explore", closing=True)
    answer_open_count = _tag_count(content, "Answer", closing=False)
    answer_close_count = _tag_count(content, "Answer", closing=True)

    has_action = action_open_count > 0 or action_close_count > 0
    has_explore = explore_open_count > 0 or explore_close_count > 0
    has_answer = answer_open_count > 0 or answer_close_count > 0

    if has_explore and not allow_explore:
        return (
            False,
            "<Explore> is unavailable in this workflow. Use <Action> to inspect "
            "local data or <Answer> to complete the task.",
        )

    response_block_count = sum(1 for present in (has_action, has_explore, has_answer) if present)
    if response_block_count == 0:
        if not allow_explore:
            return False, "Missing <Action>...</Action> or <Answer>...</Answer> block."
        return (
            False,
            "Missing <Action>...</Action>, <Explore>...</Explore>, "
            "or <Answer>...</Answer> block.",
        )
    if response_block_count > 1:
        return (
            False,
            "Reply must contain exactly one of <Action>...</Action>, "
            "<Explore>...</Explore>, or <Answer>...</Answer>.",
        )

    if has_action and (action_open_count != 1 or action_close_count != 1):
        return False, "Reply must contain exactly one <Action>...</Action> block."
    if has_explore and (explore_open_count != 1 or explore_close_count != 1):
        return False, "Reply must contain exactly one <Explore>...</Explore> block."
    if has_answer and (answer_open_count != 1 or answer_close_count != 1):
        return False, "Reply must contain exactly one <Answer>...</Answer> block."

    if has_action:
        action = extract_action_block(content)
        if not action:
            return False, "<Action> cannot be empty."
        return True, None

    if has_explore:
        explore = extract_explore_block(content)
        if not explore:
            return False, "<Explore> cannot be empty."
        if "```" in explore:
            return False, "<Explore> cannot contain a code block."
        return True, None

    answer = extract_answer_block(content)
    if not answer:
        return False, "<Answer> cannot be empty."
    if "```" in answer:
        return False, "<Answer> cannot contain a code block."
    return True, None


def _repair_response_shell(content: str) -> tuple[str, str] | None:
    """Repair one unambiguous response block missing one boundary tag."""
    if _has_tag(content, "Final Answer"):
        return None

    response_tags = ("Action", "Explore", "Answer")
    present_tags = [tag for tag in response_tags if _has_tag(content, tag)]
    if len(present_tags) != 1:
        return None

    tag = present_tags[0]
    open_count = _tag_count(content, tag, closing=False)
    close_count = _tag_count(content, tag, closing=True)
    if open_count == 1 and close_count == 0:
        repaired = content.rstrip() + f"\n</{tag}>"
        reason = f"added missing </{tag}> closing tag"
    elif open_count == 0 and close_count == 1:
        opening_index = _missing_opening_insertion_index(content, tag)
        if opening_index is None:
            return None
        repaired = content[:opening_index] + f"<{tag}>" + content[opening_index:]
        reason = f"added missing <{tag}> opening tag"
    else:
        return None

    is_valid, _ = validate_turn_structure(repaired, allow_explore=True)
    if not is_valid:
        return None
    return repaired, reason


def _missing_opening_insertion_index(content: str, tag: str) -> int | None:
    """Locate a conservative insertion point for one missing response opener."""
    close_match = re.search(rf"</{re.escape(tag)}>", content, flags=re.IGNORECASE)
    if close_match is None:
        return None

    prefix = content[: close_match.start()]
    think_closes = list(re.finditer(r"</Think>", prefix, flags=re.IGNORECASE))
    if think_closes:
        return think_closes[-1].end()

    if tag.casefold() == "action":
        code_match = re.search(r"```python\b", prefix, flags=re.IGNORECASE)
        if code_match is None:
            return None
        return code_match.start()

    return len(prefix) - len(prefix.lstrip())


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


def wrap_observation(observation: str) -> str:
    return f"<Observation>\n{observation}\n</Observation>"


def wrap_feedback(feedback: str) -> str:
    return f"<Feedback>\n{feedback}\n</Feedback>"


def build_protocol_error_feedback(
    reason: Optional[str],
    *,
    allow_explore: bool = True,
) -> str:
    detail = reason or "Malformed assistant reply."
    formats = "Optional: <Think>...</Think>\n<Action>...</Action>\nor:\n"
    if allow_explore:
        formats += "Optional: <Think>...</Think>\n<Explore>...</Explore>\nor:\n"
    formats += "<Answer>...</Answer>\n"
    explore_guidance = (
        "Use <Explore> only for a plain-text perception request. "
        if allow_explore
        else ""
    )
    return (
        f"Protocol error: {detail}\n"
        "Reply using the strict format:\n"
        f"{formats}"
        "Always close every tag explicitly. In particular, finish completion replies with </Answer>.\n"
        "Use <Action> only for exactly one fenced ```python``` block. "
        f"{explore_guidance}"
        "Use <Answer> only for plain-text completion after the task is done."
    )


def format_observation(exec_result: ExecutionResult) -> str:
    if exec_result.success:
        return exec_result.stdout or "(no output)"
    return exec_result.stderr or exec_result.stdout or "(execution failed with no output)"


def truncate_observation(
    text: str,
    *,
    obs_max_tokens: int,
    obs_head_tokens: int,
    obs_tail_tokens: int,
) -> str:
    """Keep at most obs_max_tokens tokens, preserving head and tail."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) < obs_max_tokens:
            return text
        head = enc.decode(tokens[:obs_head_tokens])
        tail = enc.decode(tokens[-obs_tail_tokens:])
        return head + "\n...\n" + tail
    except Exception:
        max_chars = obs_max_tokens * 4
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]


__all__ = [
    "NormalizedReActReply",
    "ReActTurnResult",
    "build_execution_message",
    "build_perception_message",
    "build_protocol_error_feedback",
    "extract_action_block",
    "extract_answer_block",
    "extract_explore_block",
    "extract_final_answer_from_action",
    "extract_strict_python_from_action",
    "extract_think_block",
    "normalize_react_reply",
    "parse_react_reply",
    "truncate_observation",
    "validate_turn_structure",
    "wrap_feedback",
    "wrap_observation",
]
