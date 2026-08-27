"""Deterministic reward for the raw DSLighting ReAct response protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from dslighting.workflows.search.react.perception_protocol import (
    validate_strict_perception_reply,
)
from dslighting.workflows.search.react.protocol import validate_strict_react_reply

AgentRole = Literal["solver", "perception"]


@dataclass(frozen=True)
class ModelProtocolTurn:
    role: AgentRole
    content: str


@dataclass(frozen=True)
class ProtocolScore:
    value: float
    valid_turns: int
    total_turns: int
    terminal_answer_valid: bool
    solver_valid_turns: int = 0
    solver_total_turns: int = 0
    perception_valid_turns: int = 0
    perception_total_turns: int = 0


def infer_agent_role_from_request(request: Mapping[str, Any]) -> AgentRole:
    """Infer the prompt-conditioned role of one captured model request."""
    messages = request.get("messages") or []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        content = str(message.get("content") or "")
        if "You are a Perception Agent" in content:
            return "perception"
        if "You are a Solving Agent" in content:
            return "solver"
    # Solver-only historical events may not contain the compacted request.
    return "solver"


def model_protocol_turns_from_events(
    events: Sequence[dict[str, Any]],
) -> list[ModelProtocolTurn]:
    """Extract successful raw replies together with their prompt role."""
    turns: list[ModelProtocolTurn] = []
    for event in events:
        if event.get("event_type") != "model_request":
            continue
        data = event.get("data") or {}
        if data.get("status", "ok") != "ok":
            continue
        response = data.get("response") or {}
        choices = response.get("choices") or []
        if not choices:
            continue
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            turns.append(
                ModelProtocolTurn(
                    role=infer_agent_role_from_request(data.get("request") or {}),
                    content=content,
                )
            )
    return turns


def model_responses_from_events(events: Sequence[dict[str, Any]]) -> list[str]:
    """Extract successful assistant response text from raw AGL events."""
    return [turn.content for turn in model_protocol_turns_from_events(events)]


def score_protocol_responses(responses: Sequence[str]) -> ProtocolScore:
    """Score strict turn validity plus a valid terminal ``<Answer>`` reply."""
    return score_protocol_turns(
        [ModelProtocolTurn(role="solver", content=response) for response in responses],
        perception_enabled=False,
    )


def score_protocol_turns(
    turns: Sequence[ModelProtocolTurn],
    *,
    perception_enabled: bool,
) -> ProtocolScore:
    """Score raw Solver and Perception replies with their respective protocols."""
    validity = []
    for turn in turns:
        if turn.role == "perception":
            valid = perception_enabled and validate_strict_perception_reply(turn.content)[0]
        else:
            valid = validate_strict_react_reply(
                turn.content,
                allow_explore=perception_enabled,
            )[0]
        validity.append(valid)

    total_turns = len(validity)
    valid_turns = sum(validity)
    valid_rate = valid_turns / total_turns if total_turns else 0.0
    solver_indices = [index for index, turn in enumerate(turns) if turn.role == "solver"]
    solver_valid_turns = sum(validity[index] for index in solver_indices)
    perception_indices = [index for index, turn in enumerate(turns) if turn.role == "perception"]
    perception_valid_turns = sum(validity[index] for index in perception_indices)
    last_solver_index = solver_indices[-1] if solver_indices else None
    terminal_answer_valid = bool(
        last_solver_index is not None
        and validity[last_solver_index]
        and re.search(r"</Think>\s*<Answer>", turns[last_solver_index].content)
    )
    value = 0.8 * valid_rate + 0.2 * float(terminal_answer_valid)
    return ProtocolScore(
        value=value,
        valid_turns=valid_turns,
        total_turns=total_turns,
        terminal_answer_valid=terminal_answer_valid,
        solver_valid_turns=solver_valid_turns,
        solver_total_turns=len(solver_indices),
        perception_valid_turns=perception_valid_turns,
        perception_total_turns=len(perception_indices),
    )


def combine_rewards(
    task_reward: float,
    protocol_reward: float,
    *,
    protocol_weight: float = 0.1,
) -> float:
    if not 0.0 <= protocol_weight <= 1.0:
        raise ValueError("protocol_weight must be between 0 and 1")
    return (1.0 - protocol_weight) * task_reward + protocol_weight * protocol_reward


__all__ = [
    "AgentRole",
    "ModelProtocolTurn",
    "ProtocolScore",
    "combine_rewards",
    "infer_agent_role_from_request",
    "model_protocol_turns_from_events",
    "model_responses_from_events",
    "score_protocol_responses",
    "score_protocol_turns",
]
