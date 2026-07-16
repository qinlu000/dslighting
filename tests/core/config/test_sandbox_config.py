from __future__ import annotations

import pytest

from dslighting.core import ConfigBuilder
from dslighting.error import ConfigurationError


def test_build_config_accepts_strict_local_sandbox_block() -> None:
    config = ConfigBuilder().build_config(
        model="openai/test-model",
        sandbox={
            "local_isolation": "bubblewrap",
            "environment_policy": "allowlist",
            "network_policy": "disabled",
        },
    )

    assert config.sandbox.local_isolation == "bubblewrap"
    assert config.sandbox.environment_policy == "allowlist"
    assert config.sandbox.network_policy == "disabled"


@pytest.mark.parametrize(
    "sandbox",
    [
        "bubblewrap",
        {"local_isolation": "unknown"},
        {"environment_policy": "unknown"},
        {"network_policy": "unknown"},
    ],
)
def test_build_config_rejects_invalid_sandbox_block(sandbox: object) -> None:
    with pytest.raises((ConfigurationError, ValueError)):
        ConfigBuilder().build_config(model="openai/test-model", sandbox=sandbox)  # type: ignore[arg-type]
