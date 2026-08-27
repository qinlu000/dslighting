from __future__ import annotations

import pytest

from dslighting.workflows.search.react.perception_protocol import (
    PerceptionTurnResult,
    parse_perception_reply,
    validate_strict_perception_reply,
)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "<Think>Inspect rows.</Think>" "<Action>```python\nprint('rows')\n```</Action>",
            PerceptionTurnResult(action_code="print('rows')"),
        ),
        (
            "<Think>The request is answered.</Think>" "<Report>rows=3; missing=0</Report>",
            PerceptionTurnResult(report="rows=3; missing=0"),
        ),
        (
            "<Think>Inspect rows.</Think>" "```python\nprint('rows')\n```</Action>",
            PerceptionTurnResult(action_code="print('rows')"),
        ),
        (
            "<Think>The request is answered.</Think>" "<Report>rows=3; missing=0",
            PerceptionTurnResult(report="rows=3; missing=0"),
        ),
    ],
)
def test_parse_perception_reply(
    reply: str,
    expected: PerceptionTurnResult,
) -> None:
    assert parse_perception_reply(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "rows=3",
        "<Action>```python\nprint('rows')\n```</Action>",
        "  <Think>Inspect.</Think><Action>```python\nprint('rows')\n```</Action>",
        "<think>Inspect.</think><Action>```python\nprint('rows')\n```</Action>",
        "<Action>print('rows')</Action>",
        "<Action>```python\nprint('rows')\n```</Action><Report>rows=3</Report>",
        "<Think>Inspect.</Think>extra<Action>```python\nprint('rows')\n```</Action>",
    ],
)
def test_parse_perception_reply_rejects_invalid_input(reply: str) -> None:
    assert parse_perception_reply(reply).next_user_message is not None


def test_strict_perception_validation_rejects_runtime_repair() -> None:
    valid = "<Think>Done.</Think><Report>rows=3</Report>"
    repairable = "<Think>Done.</Think><Report>rows=3"

    assert validate_strict_perception_reply(valid) == (True, None)
    assert validate_strict_perception_reply(repairable)[0] is False
