from __future__ import annotations

import pytest

from dslighting.workflows.search.react.perception_protocol import (
    normalize_perception_reply,
    parse_perception_reply,
)


def test_parse_perception_action() -> None:
    result = parse_perception_reply(
        "<Action>```python\nprint('rows')\n```</Action>"
    )

    assert result.action_code == "print('rows')"
    assert result.report is None
    assert result.next_user_message is None


def test_parse_perception_report() -> None:
    result = parse_perception_reply("<Report>rows=3; missing=0</Report>")

    assert result.report == "rows=3; missing=0"
    assert result.action_code is None
    assert result.next_user_message is None


def test_parse_perception_accepts_case_insensitive_tags() -> None:
    result = parse_perception_reply(
        "<action>```python\nprint('rows')\n```</ACTION>",
    )

    assert result.action_code == "print('rows')"
    assert result.next_user_message is None


def test_parse_perception_repairs_one_unclosed_action() -> None:
    reply = "<Action>```python\nprint('rows')\n```"

    normalized = normalize_perception_reply(reply)
    result = parse_perception_reply(reply)

    assert normalized.repaired is True
    assert normalized.normalized_content.endswith("</Action>")
    assert result.action_code == "print('rows')"
    assert result.next_user_message is None


def test_parse_perception_repairs_one_unclosed_report() -> None:
    result = parse_perception_reply("<report>rows=3; missing=0")

    assert result.report == "rows=3; missing=0"
    assert result.next_user_message is None


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "rows=3",
        "<Think>Inspect.</Think><Action>```python\nprint('rows')\n```</Action>",
        "<Action>print('rows')</Action>",
        "<Action>```python\nprint('rows')\n```</Action><Report>rows=3</Report>",
        "<Report></Report>",
        "<Report>```python\nprint('rows')\n```</Report>",
        "<Explore>Delegate again.</Explore>",
        "extra<Report>rows=3</Report>",
    ],
)
def test_parse_perception_rejects_invalid_reply(reply: str) -> None:
    result = parse_perception_reply(reply)

    assert result.action_code is None
    assert result.report is None
    assert result.next_user_message is not None
    assert "Perception protocol error:" in result.next_user_message
    assert "<Think>" in result.next_user_message
