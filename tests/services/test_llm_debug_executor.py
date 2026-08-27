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
from dslighting.services.llm.observed_call import completion_with_observability
from dslighting.services.llm.service import LLMService
from dslighting.services.llm.executor import MAX_TRANSPORT_RETRY_DELAY_SECONDS


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
            request_timeout_seconds=15,
            sdk_max_retries=1,
        )
    )
    disabled_kwargs = disabled._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert disabled_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert disabled_kwargs["timeout"] == 15
    assert disabled_kwargs["num_retries"] == 1
    assert "temperature" not in disabled_kwargs

    local_vllm = LLMService(
        LLMConfig(
            model="openai/local-qwen",
            api_key="secret",
            thinking=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    local_kwargs = local_vllm._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert local_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }


def test_bad_request_is_not_retried() -> None:
    GlobalAPIKeyPool.clear_pools()
    service = LLMService(
        LLMConfig(
            model="custom-deployment",
            provider="openai",
            api_key="secret",
            api_base="https://example.com/v1",
        )
    )
    error = litellm_exceptions.BadRequestError(
        message="LLM Provider NOT provided",
        model="custom-deployment",
        llm_provider="openai",
    )

    assert service._classify_error_action(error) == "fail_fast"

    default_disabled = LLMService(
        LLMConfig(model="gpt-test", api_key="secret", api_base="https://example.com/v1")
    )
    default_disabled_kwargs = default_disabled._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert default_disabled_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    inherited = LLMService(
        LLMConfig(
            model="gpt-test",
            api_key="secret",
            api_base="https://example.com/v1",
            thinking=None,
        )
    )
    inherited_kwargs = inherited._build_completion_kwargs(
        messages=[{"role": "user", "content": "Return text"}],
        response_format=None,
        api_key="secret",
    )
    assert "extra_body" not in inherited_kwargs


def test_observed_completion_uses_llm_config_transport_settings(monkeypatch) -> None:
    captured: dict = {}

    def _fake_completion(**kwargs):
        captured.update(kwargs)
        return _Response("ok")

    monkeypatch.setattr("litellm.completion", _fake_completion)
    config = LLMConfig(
        model="gpt-test",
        api_key="secret",
        api_base="https://example.com/v1",
        thinking=False,
        request_timeout_seconds=20,
        sdk_max_retries=2,
    )

    completion_with_observability(
        llm_config=config,
        messages=[{"role": "user", "content": "Return text"}],
    )

    assert captured["timeout"] == 20
    assert captured["num_retries"] == 2
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "temperature" not in captured


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
async def test_reasoning_content_is_used_when_content_is_empty(
    monkeypatch,
) -> None:
    GlobalAPIKeyPool.clear_pools()
    response = _Response("")
    response.id = "response-123"
    response.model = "gpt-test"
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.reasoning_content = "final response from reasoning content"
    response.choices[0].message.tool_calls = [{"id": "call-1"}]

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return response

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    service = LLMService(
        LLMConfig(
            model="gpt-test",
            api_key="secret",
            api_base="https://example.com/v1",
            max_retries=1,
        )
    )
    result = await service.call("Return text")

    assert result == "final response from reasoning content"
    assert response.choices[0].message.content == result


@pytest.mark.asyncio
async def test_call_with_json_emits_single_logical_call_with_validation_retry(monkeypatch, tmp_path: Path) -> None:
    GlobalAPIKeyPool.clear_pools()
    session = init_debug(enabled=True, profile="full", output_dir=str(tmp_path), console_output=False)

    responses = iter([_Response('{"value": '), _Response('{"value": "ok"}')])

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return next(responses)

    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)

    service = LLMService(
        LLMConfig(
            model="gpt-test",
            api_key="secret",
            api_base="https://example.com/v1",
            max_retries=2,
        )
    )
    try:
        result = await service.call_with_json("Return JSON", _OutputModel)
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
            max_retries=1,
        )
    )
    result = await service.call_with_json("Return JSON", _OutputModel)

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
            max_retries=1,
        )
    )

    with pytest.raises(LLMServiceError, match="exhausting 2 API key"):
        await service.call_with_json("Return JSON", _OutputModel)

    assert attempted_keys == ["bad-key-1", "bad-key-2"]
