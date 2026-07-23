from __future__ import annotations

from pathlib import Path

import pytest

from dslighting.config import DSLightingConfig
from dslighting.error import ConfigurationError
from dslighting.services.sandbox_backends.backends.docker import DockerSandboxBackend
from dslighting.services.workspace import WorkspaceService
from dslighting.workflows.factory.builtin import _create_sandbox_service


def test_factory_wires_docker_resource_and_image_settings(tmp_path: Path) -> None:
    config = DSLightingConfig(
        sandbox={
            "backend": "docker",
            "docker_image": "agenticdatabench:test",
            "environment_policy": "allowlist",
            "network_policy": "disabled",
            "memory_mb": 6144,
            "cpu_cores": 3.0,
            "pids_limit": 128,
        }
    )
    workspace = WorkspaceService("docker-factory-test", base_dir=str(tmp_path))

    service = _create_sandbox_service(workspace, config)

    assert isinstance(service.backend, DockerSandboxBackend)
    assert service.backend.image == "agenticdatabench:test"
    assert service.backend.config.memory_mb == 6144
    assert service.backend.config.cpu_cores == 3.0
    assert service.backend.pids_limit == 128
    service._executor.shutdown(wait=True)


def test_factory_requires_docker_image(tmp_path: Path) -> None:
    config = DSLightingConfig(sandbox={"backend": "docker"})
    workspace = WorkspaceService("docker-factory-test", base_dir=str(tmp_path))

    with pytest.raises(ConfigurationError, match="docker_image"):
        _create_sandbox_service(workspace, config)
