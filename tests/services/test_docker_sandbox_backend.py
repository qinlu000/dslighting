from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dslighting.services.sandbox_backends.backends.base import SandboxBackendConfig
from dslighting.services.sandbox_backends.backends.docker import DockerSandboxBackend


class _FakeImages:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get(self, image: str) -> object:
        self.requested.append(image)
        return object()


class _FakeContainer:
    id = "fake-container-id"

    def __init__(self, run_kwargs: dict) -> None:
        self.run_kwargs = run_kwargs
        self.removed = False
        self.exec_calls: list[dict] = []

    def exec_run(self, command: list[str], **kwargs):
        self.exec_calls.append({"command": command, **kwargs})
        workspace = next(
            Path(source)
            for source, mount in self.run_kwargs["volumes"].items()
            if mount["bind"] == "/workspace"
        )
        script = workspace / Path(command[1]).relative_to("/workspace")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
        return SimpleNamespace(
            exit_code=completed.returncode,
            output=(completed.stdout, completed.stderr),
        )

    def remove(self, *, force: bool) -> None:
        assert force is True
        self.removed = True


class _FakeContainers:
    def __init__(self) -> None:
        self.run_calls: list[dict] = []
        self.instances: list[_FakeContainer] = []

    def run(self, **kwargs) -> _FakeContainer:
        self.run_calls.append(kwargs)
        container = _FakeContainer(kwargs)
        self.instances.append(container)
        return container


class _FakeDockerClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.closed = False
        self.pinged = False

    def ping(self) -> bool:
        self.pinged = True
        return True

    def close(self) -> None:
        self.closed = True


def _backend(client: _FakeDockerClient) -> DockerSandboxBackend:
    return DockerSandboxBackend(
        image="agenticdatabench:test",
        config=SandboxBackendConfig(
            timeout=5,
            memory_mb=2048,
            cpu_cores=1.5,
            environment_policy="allowlist",
            network_policy="disabled",
            env_vars={"MPLBACKEND": "Agg"},
        ),
        client_factory=lambda: client,
    )


@pytest.mark.asyncio
async def test_reuses_task_container_and_persists_workspace(tmp_path: Path) -> None:
    public = tmp_path / "dataset" / "public"
    public.mkdir(parents=True)
    source = public / "input.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.csv").symlink_to(source)
    client = _FakeDockerClient()
    backend = _backend(client)

    first = await backend.execute(
        "from pathlib import Path; Path('state.txt').write_text(Path('input.csv').read_text()); print('one')",
        str(workspace),
    )
    second = await backend.execute(
        "from pathlib import Path; print(Path('state.txt').read_text().strip())",
        str(workspace),
    )

    assert first.success and first.stdout.strip() == "one"
    assert second.success and second.stdout.strip() == "value\n1"
    assert len(client.containers.run_calls) == 1
    run_kwargs = client.containers.run_calls[0]
    assert run_kwargs["network_mode"] == "none"
    assert run_kwargs["mem_limit"] == "2048m"
    assert run_kwargs["nano_cpus"] == 1_500_000_000
    assert run_kwargs["volumes"][str(workspace.resolve())] == {
        "bind": "/workspace",
        "mode": "rw",
    }
    assert run_kwargs["volumes"][str(source.resolve())]["mode"] == "ro"
    assert first.metadata["backend"] == "docker"
    assert first.metadata["frozen_input_mount_count"] == 1

    container = client.containers.instances[0]
    await backend.shutdown()
    assert container.removed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_rejects_credentials_before_connecting() -> None:
    client = _FakeDockerClient()
    backend = DockerSandboxBackend(
        image="agenticdatabench:test",
        config=SandboxBackendConfig(env_vars={"OPENAI_API_KEY": "secret"}),
        client_factory=lambda: client,
    )

    with pytest.raises(ValueError, match="forbidden"):
        await backend.initialize()

    assert client.pinged is False


@pytest.mark.asyncio
async def test_backend_instance_cannot_switch_workspaces(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    backend = _backend(_FakeDockerClient())

    assert (await backend.execute("print('ok')", str(first_workspace))).success
    with pytest.raises(ValueError, match="cannot switch"):
        await backend.execute("print('no')", str(second_workspace))

    await backend.shutdown()
