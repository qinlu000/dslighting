from __future__ import annotations

import pytest

from dslighting.workflows.search.react.perception_protocol import (
    PerceptionTurnResult,
    parse_perception_reply,
)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "<Think>Inspect rows.</Think>"
            "<Action>```python\nprint('rows')\n```</Action>",
            PerceptionTurnResult(action_code="print('rows')"),
        ),
        (
            "<Think>The request is answered.</Think>"
            "<Report>rows=3; missing=0</Report>",
            PerceptionTurnResult(report="rows=3; missing=0"),
        ),
        (
            "<Action>```python\nprint('rows')\n```</Action>",
            PerceptionTurnResult(action_code="print('rows')"),
        ),
        (
            "<Think>Inspect rows.</Think>"
            "```python\nprint('rows')\n```</Action>",
            PerceptionTurnResult(action_code="print('rows')"),
        ),
        (
            "<Think>The request is answered.</Think>"
            "<Report>rows=3; missing=0",
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
        "<Action>print('rows')</Action>",
        "<Action>```python\nprint('rows')\n```</Action><Report>rows=3</Report>",
        "<Think>Inspect.</Think>extra<Action>```python\nprint('rows')\n```</Action>",
    ],
)
def test_parse_perception_reply_rejects_invalid_input(reply: str) -> None:
    assert parse_perception_reply(reply).next_user_message is not None
