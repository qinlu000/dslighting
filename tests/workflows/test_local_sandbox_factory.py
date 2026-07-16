from __future__ import annotations

from pathlib import Path

from dslighting.config import DSLightingConfig, SandboxConfig
from dslighting.services.sandbox_backends.backends.local import LocalSandboxBackend
from dslighting.services.workspace import WorkspaceService
from dslighting.workflows.factory.builtin import _create_sandbox_service


def test_sandbox_config_compatibility_defaults() -> None:
    config = SandboxConfig()

    assert config.local_isolation == "process"
    assert config.environment_policy == "inherit"
    assert config.network_policy == "inherit"


def test_factory_wires_strict_local_isolation_settings(tmp_path: Path) -> None:
    config = DSLightingConfig(
        sandbox={
            "local_isolation": "bubblewrap",
            "environment_policy": "allowlist",
            "network_policy": "disabled",
        },
        run={"parameters": {"sandbox_env": {"EXPLICIT_SAFE_VALUE": "visible"}}},
    )
    workspace = WorkspaceService("factory-test", base_dir=str(tmp_path))

    service = _create_sandbox_service(workspace, config)

    assert isinstance(service.backend, LocalSandboxBackend)
    assert service.backend.config.isolation == "bubblewrap"
    assert service.backend.config.environment_policy == "allowlist"
    assert service.backend.config.network_policy == "disabled"
    assert service.backend.config.env_vars["EXPLICIT_SAFE_VALUE"] == "visible"
    service._executor.shutdown(wait=True)
