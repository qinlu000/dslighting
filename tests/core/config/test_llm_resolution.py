from __future__ import annotations

import json

import pytest

from dslighting.config import LLMConfig
from dslighting.core import ConfigBuilder
from dslighting.core.config.llm_resolution import build_llm_config
from dslighting.error import ConfigurationError


def test_global_llm_request_timeout_defaults_to_five_minutes() -> None:
    assert LLMConfig().request_timeout_seconds == 300.0


def test_model_override_beats_global_env(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "global-key")
    monkeypatch.setenv("API_BASE", "https://global.example/v1")
    monkeypatch.setenv(
        "LLM_MODEL_CONFIGS",
        json.dumps(
            {
                "model-a": {
                    "api_key": ["key-1", "key-2"],
                    "api_base": "https://override.example/v1",
                    "provider": "siliconflow",
                    "temperature": 0.1,
                }
            }
        ),
    )

    config = ConfigBuilder().build_config(model="model-a")

    assert config.llm.model == "model-a"
    assert config.llm.api_key is None
    assert config.llm.api_keys == ["key-1", "key-2"]
    assert config.llm.api_base == "https://override.example/v1"
    assert config.llm.provider == "siliconflow"
    assert config.llm.temperature == pytest.approx(0.1)


def test_explicit_params_beat_model_override(monkeypatch) -> None:
    monkeypatch.setenv(
        "LLM_MODEL_CONFIGS",
        json.dumps(
            {
                "model-a": {
                    "api_key": ["key-1", "key-2"],
                    "api_base": "https://override.example/v1",
                    "provider": "siliconflow",
                    "temperature": 0.1,
                }
            }
        ),
    )

    config = ConfigBuilder().build_config(
        model="model-a",
        api_key="manual-key",
        api_base="https://manual.example/v1",
        provider="manual-provider",
        temperature=0.9,
    )

    assert config.llm.model == "model-a"
    assert config.llm.api_key == "manual-key"
    assert config.llm.api_keys is None
    assert config.llm.api_base == "https://manual.example/v1"
    assert config.llm.provider == "manual-provider"
    assert config.llm.temperature == pytest.approx(0.9)


def test_api_key_list_is_normalized_to_api_keys() -> None:
    config = ConfigBuilder().build_config(model="model-a", api_key=["key-1", "key-2"])

    assert config.llm.api_key is None
    assert config.llm.api_keys == ["key-1", "key-2"]


def test_openai_api_key_is_only_a_global_fallback(monkeypatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert build_llm_config(model="openai-fallback-test-model").api_key == "openai-key"

    monkeypatch.setenv("API_KEY", "primary-key")
    assert build_llm_config(model="openai-fallback-test-model").api_key == "primary-key"


def test_custom_api_base_defaults_to_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE", "https://gateway.example/v1")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    config = build_llm_config(model="custom-deployment", api_key="key")

    assert config.provider == "openai"


def test_explicit_provider_beats_openai_compatible_default(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE", "https://gateway.example/v1")

    config = build_llm_config(
        model="custom-deployment",
        api_key="key",
        provider="anthropic",
    )

    assert config.provider == "anthropic"


def test_build_config_rejects_conflicting_api_key_and_api_keys() -> None:
    with pytest.raises(ConfigurationError, match="Only one of `api_key` or `api_keys`"):
        ConfigBuilder().build_config(model="model-a", api_key="k1", api_keys=["k2"])


def test_builder_accepts_one_global_llm_config() -> None:
    llm_config = build_llm_config(
        model="model-a",
        api_key="key-a",
        thinking=False,
        max_retries=3,
        request_timeout_seconds=45,
        sdk_max_retries=0,
        max_concurrent_per_key=8,
        global_max_concurrency=6,
    )

    config = ConfigBuilder().build_config(workflow="react", llm_config=llm_config)

    assert config.llm == llm_config
    assert config.llm is not llm_config


def test_builder_rejects_global_and_local_llm_config_together() -> None:
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        ConfigBuilder().build_config(
            model="local-model",
            llm_config=LLMConfig(model="global-model"),
        )
