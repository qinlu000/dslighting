"""Thin ReAct protocol operator.

This operator parses ReAct response actions and formats execution observations.
It does not execute code directly; workflows should delegate execution to the
shared execute operator.
"""

from __future__ import annotations

from typing import Optional

from dslighting.ops.base import Operator
from dslighting.workflows.search.react.protocol import (
    ReActTurnResult,
    build_execution_message,
    build_perception_message,
    parse_react_reply,
)
from dslighting.workflows.search.react.validation import (
    validate_react_operator_params,
)


class ReActOperator(Operator):
    """Parse ReAct protocol turns and render execution observations."""

    def __init__(
        self,
        llm_service=None,
        sandbox_service=None,
        max_steps: int = 10,
        obs_max_tokens: int = 4000,
        obs_head_tokens: int = 2000,
        obs_tail_tokens: int = 2000,
    ) -> None:
        super().__init__(llm_service=llm_service, name="ReAct")
        _ = sandbox_service
        validate_react_operator_params(
            obs_max_tokens=obs_max_tokens,
            obs_head_tokens=obs_head_tokens,
            obs_tail_tokens=obs_tail_tokens,
        )
        self.max_steps = max_steps
        self.obs_max_tokens = obs_max_tokens
        self.obs_head_tokens = obs_head_tokens
        self.obs_tail_tokens = obs_tail_tokens

    async def __call__(
        self,
        assistant_reply: str,
        *,
        expected_output_filename: Optional[str] = None,
        allow_explore: bool = False,
    ) -> ReActTurnResult:
        _ = expected_output_filename
        return parse_react_reply(
            assistant_reply,
            allow_explore=allow_explore,
        )

    def build_execution_message(
        self,
        exec_result,
        *,
        critical_footer: Optional[str] = None,
        escape_output: bool = False,
    ) -> str:
        return build_execution_message(
            exec_result,
            obs_max_tokens=self.obs_max_tokens,
            obs_head_tokens=self.obs_head_tokens,
            obs_tail_tokens=self.obs_tail_tokens,
            critical_footer=critical_footer,
            escape_output=escape_output,
        )

    def build_perception_message(self, perception_result: str) -> str:
        return build_perception_message(
            perception_result,
            obs_max_tokens=self.obs_max_tokens,
            obs_head_tokens=self.obs_head_tokens,
            obs_tail_tokens=self.obs_tail_tokens,
        )


__all__ = ["ReActOperator", "ReActTurnResult"]
