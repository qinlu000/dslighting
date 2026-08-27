from __future__ import annotations

import pytest

from dslighting.workflows.search.react.protocol import validate_strict_react_reply
from experiments.training_mix.protocol_reward import (
    ModelProtocolTurn,
    combine_rewards,
    model_protocol_turns_from_events,
    model_responses_from_events,
    score_protocol_responses,
    score_protocol_turns,
)

VALID_ACTION = """<Think>Inspect the input first.</Think>
<Action>
```python
print("ok")
```
</Action>"""

VALID_ANSWER = """<Think>The artifact is complete.</Think>
<Answer>Completed and verified the requested output.</Answer>"""

VALID_EXPLORE = """<Think>I need a focused summary.</Think>
<Explore>Inspect missingness and target balance.</Explore>"""

VALID_PERCEPTION_ACTION = """<Think>Inspect the requested columns.</Think>
<Action>
```python
print("summary")
```
</Action>"""

VALID_REPORT = """<Think>The requested evidence is sufficient.</Think>
<Report>The target is balanced and feature_x has 3 missing values.</Report>"""


@pytest.mark.parametrize(
    "response",
    [
        VALID_ACTION,
        VALID_ANSWER,
    ],
)
def test_strict_protocol_accepts_prompt_compliant_raw_responses(
    response: str,
) -> None:
    assert validate_strict_react_reply(response, allow_explore=False) == (True, None)


@pytest.mark.parametrize(
    "response",
    [
        "<Action>\n```python\nprint('missing think')\n```\n</Action>",
        "prefix <Think>reason</Think><Answer>done</Answer>",
        "<Think></Think><Answer>done</Answer>",
        "<Think>reason</Think><Action>print('no fence')</Action>",
        "<Think>reason</Think><Action>\n```python\n\n```\n</Action>",
        "<Think>reason</Think><Action>\n```python\nprint('unclosed')\n</Action>",
        "<Think>reason</Think><Answer>```python\nprint('no')\n```</Answer>",
        "<Think>reason</Think><Explore>inspect data</Explore>",
    ],
)
def test_strict_protocol_rejects_malformed_raw_responses(response: str) -> None:
    is_valid, reason = validate_strict_react_reply(response, allow_explore=False)

    assert is_valid is False
    assert reason


def test_protocol_score_uses_turn_rate_and_terminal_answer() -> None:
    malformed_repairable = (
        "<Think>Try code.</Think>" "<Action>\n```python\nprint('repairable by runtime')\n```"
    )

    score = score_protocol_responses([VALID_ACTION, malformed_repairable, VALID_ANSWER])

    assert score.valid_turns == 2
    assert score.total_turns == 3
    assert score.terminal_answer_valid is True
    assert score.value == pytest.approx(0.8 * (2 / 3) + 0.2)


def test_model_response_extraction_ignores_failed_requests() -> None:
    events = [
        {
            "event_type": "model_request",
            "data": {
                "status": "error",
                "response": {"choices": [{"message": {"content": "bad"}}]},
            },
        },
        {
            "event_type": "model_request",
            "data": {
                "status": "ok",
                "response": {"choices": [{"message": {"content": VALID_ANSWER}}]},
            },
        },
        {"event_type": "reward", "data": {"value": 1.0}},
    ]

    assert model_responses_from_events(events) == [VALID_ANSWER]


def test_role_aware_protocol_score_keeps_solver_and_perception_separate() -> None:
    turns = [
        ModelProtocolTurn("solver", VALID_EXPLORE),
        ModelProtocolTurn("perception", VALID_PERCEPTION_ACTION),
        ModelProtocolTurn("perception", VALID_REPORT),
        ModelProtocolTurn("solver", VALID_ANSWER),
    ]

    score = score_protocol_turns(turns, perception_enabled=True)

    assert score.value == pytest.approx(1.0)
    assert (score.solver_valid_turns, score.solver_total_turns) == (2, 2)
    assert (score.perception_valid_turns, score.perception_total_turns) == (2, 2)
    assert score.terminal_answer_valid is True


def test_model_turn_extraction_infers_role_from_system_prompt() -> None:
    events = [
        {
            "event_type": "model_request",
            "data": {
                "status": "ok",
                "request": {
                    "messages": [{"role": "system", "content": "You are a Perception Agent."}]
                },
                "response": {"choices": [{"message": {"content": VALID_REPORT}}]},
            },
        },
        {
            "event_type": "model_request",
            "data": {
                "status": "ok",
                "request": {
                    "messages": [{"role": "system", "content": "You are a Solving Agent."}]
                },
                "response": {"choices": [{"message": {"content": VALID_ANSWER}}]},
            },
        },
    ]

    assert model_protocol_turns_from_events(events) == [
        ModelProtocolTurn("perception", VALID_REPORT),
        ModelProtocolTurn("solver", VALID_ANSWER),
    ]


def test_combined_reward_keeps_task_outcome_dominant() -> None:
    assert combine_rewards(1.0, 0.0) == pytest.approx(0.9)
    assert combine_rewards(0.0, 1.0) == pytest.approx(0.1)
    assert combine_rewards(1.0, 1.0) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        combine_rewards(1.0, 1.0, protocol_weight=1.1)
