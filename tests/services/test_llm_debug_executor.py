from __future__ import annotations

import json
from pathlib import Path

import pytest
import litellm.exceptions as litellm_exceptions
from pydantic import BaseModel

from dslighting.config import LLMConfig
from dslighting.debug.api import get_debug_session, init_debug
from dslighting.error import LLMServiceError
from dslighting.services.llm.pool import GlobalAPIKeyPool
from dslighting.services.llm.service import LLMService
from dslighting.services.llm.executor import MAX_TRANSPORT_RETRY_DELAY_SECONDS
from dslighting.services.llm.observed_call import summarize_empty_response


class _OutputModel(BaseModel):
    value: str


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _Message:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()

    def model_dump(self) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": self.choices[0].message.content}}],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_completion_kwargs_forward_explicit_thinking_mode() -> None:
    GlobalAPIKeyPool.clear_pools()
    disabled = LLMService(
        LLMConfig(
            model="openai/deepseek-v4-flash",
            api_key="secret",
            api_base="https://example.com/v1",
            thinking=False,
        )
    )
    disabled_kwargs = disabled._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert disabled_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    inherited = LLMService(
        LLMConfig(model="gpt-test", api_key="secret", api_base="https://example.com/v1")
    )
    inherited_kwargs = inherited._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert "extra_body" not in inherited_kwargs


@pytest.mark.asyncio
async def test_default_transport_retry_uses_ten_attempts_with_capped_backoff(monkeypatch) -> None:
    GlobalAPIKeyPool.clear_pools()
    attempts = 0
    delays: list[float] = []

    async def _fake_acompletion(**kwargs):
        nonlocal attempts
        _ = kwargs
        attempts += 1
        return _Response("ok" if attempts == 10 else "")

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    monkeypatch.setattr("dslighting.services.llm.executor.asyncio.sleep", _fake_sleep)

    service = LLMService(
        LLMConfig(model="gpt-test", api_key="secret", api_base="https://example.com/v1")
    )
    result = await service.call("Return text")

    assert result == "ok"
    assert service.config.max_retries == 10
    assert attempts == 10
    assert len(delays) == 9
    assert max(delays) == MAX_TRANSPORT_RETRY_DELAY_SECONDS


@pytest.mark.asyncio
async def test_empty_response_logs_only_safe_structural_metadata(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    GlobalAPIKeyPool.clear_pools()
    session = init_debug(enabled=True, profile="full", output_dir=str(tmp_path), console_output=False)
    response = _Response("")
    response.id = "response-123"
    response.model = "gpt-test"
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.reasoning_content = "sensitive reasoning body"
    response.choices[0].message.tool_calls = [{"id": "call-1"}]

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return response

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    service = LLMService(
        LLMConfig(model="gpt-test", api_key="secret", api_base="https://example.com/v1")
    )
    try:
        with caplog.at_level("WARNING"):
            with pytest.raises(LLMServiceError, match="empty response"):
                await service.call("Return text", max_retries=1)
    finally:
        current = get_debug_session()
        if current is not None:
            await current.close()

    metadata = summarize_empty_response(response)
    assert metadata == {
        "response_type": "_Response",
        "response_id": "response-123",
        "response_model": "gpt-test",
        "choice_count": 1,
        "finish_reason": "stop",
        "content_state": "blank",
        "content_length": 0,
        "reasoning_content_present": True,
        "reasoning_content_length": len("sensitive reasoning body"),
        "tool_call_count": 1,
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_cost": None,
            "completion_tokens_cost": None,
            "total_tokens_cost": None,
        },
    }
    assert "response-123" in caplog.text
    assert "reasoning_content_present" in caplog.text
    assert "sensitive reasoning body" not in caplog.text
    assert "secret" not in caplog.text

    assert session.output_dir is not None
    events = _load_jsonl(session.output_dir / "events.jsonl")
    assert "llm.response.empty" in [entry["event_type"] for entry in events]


@pytest.mark.asyncio
async def test_call_with_json_emits_single_logical_call_with_validation_retry(monkeypatch, tmp_path: Path) -> None:
    GlobalAPIKeyPool.clear_pools()
    session = init_debug(enabled=True, profile="full", output_dir=str(tmp_path), console_output=False)

    responses = iter([_Response('{"value": '), _Response('{"value": "ok"}')])

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return next(responses)

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

    service = LLMService(LLMConfig(model="gpt-test", api_key="secret", api_base="https://example.com/v1"))
    try:
        result = await service.call_with_json("Return JSON", _OutputModel, max_retries=2)
        assert result.value == "ok"
    finally:
        current = get_debug_session()
        if current is not None:
            await current.close()

    assert session.output_dir is not None
    events = _load_jsonl(session.output_dir / "events.jsonl")
    payloads = _load_jsonl(session.output_dir / "payloads.jsonl")

    event_types = [entry["event_type"] for entry in events]
    assert "llm.call.started" in event_types
    assert "llm.request.prepared" in event_types
    assert "llm.response.received" in event_types
    assert "llm.validation.failed" in event_types
    assert "llm.retry.scheduled" in event_types
    assert "llm.call.completed" in event_types

    logical_call_ids = {entry["llm"]["logical_call_id"] for entry in events if entry.get("llm")}
    assert len(logical_call_ids) == 1

    request_payloads = [entry for entry in payloads if entry["kind"] == "request_messages"]
    assert len(request_payloads) == 1


@pytest.mark.asyncio
async def test_call_with_json_rotates_to_next_key_on_auth_failure(monkeypatch) -> None:
    GlobalAPIKeyPool.clear_pools()
    attempted_keys: list[str] = []

    async def _fake_acompletion(**kwargs):
        api_key = kwargs["api_key"]
        attempted_keys.append(api_key)
        if api_key == "bad-key":
            raise litellm_exceptions.AuthenticationError(
                message="Api key is invalid",
                llm_provider="openai",
                model="gpt-test",
            )
        return _Response('{"value": "ok"}')

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

    service = LLMService(
        LLMConfig(
            model="gpt-test",
            api_keys=["bad-key", "good-key"],
            api_base="https://example.com/v1",
        )
    )
    result = await service.call_with_json("Return JSON", _OutputModel, max_retries=1)

    assert result.value == "ok"
    assert attempted_keys == ["bad-key", "good-key"]
    assert service._key_pool._last_error_kind["bad-key"] == "AuthenticationError"
    assert service._key_pool._cooldown_until["bad-key"] > 0.0


@pytest.mark.asyncio
async def test_call_with_json_fails_after_all_keys_auth_fail(monkeypatch) -> None:
    GlobalAPIKeyPool.clear_pools()
    attempted_keys: list[str] = []

    async def _fake_acompletion(**kwargs):
        api_key = kwargs["api_key"]
        attempted_keys.append(api_key)
        raise litellm_exceptions.AuthenticationError(
            message=f"Api key is invalid: {api_key}",
            llm_provider="openai",
            model="gpt-test",
        )

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

    service = LLMService(
        LLMConfig(
            model="gpt-test",
            api_keys=["bad-key-1", "bad-key-2"],
            api_base="https://example.com/v1",
        )
    )

    with pytest.raises(LLMServiceError, match="exhausting 2 API key"):
        await service.call_with_json("Return JSON", _OutputModel, max_retries=1)

    assert attempted_keys == ["bad-key-1", "bad-key-2"]
