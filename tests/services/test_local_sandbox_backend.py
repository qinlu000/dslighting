from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

import pytest

from dslighting.services.sandbox_backends.backends.base import SandboxBackendConfig
from dslighting.services.sandbox_backends.backends.local import LocalSandboxBackend

BWRAP_AVAILABLE = shutil.which("bwrap") is not None
requires_bwrap = pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bubblewrap is unavailable")


def _strict_backend(**kwargs: object) -> LocalSandboxBackend:
    config = SandboxBackendConfig(
        timeout=10,
        isolation="bubblewrap",
        environment_policy="allowlist",
        network_policy="disabled",
        **kwargs,
    )
    return LocalSandboxBackend(config=config)


@pytest.mark.asyncio
async def test_process_inherit_remains_the_compatibility_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("SANDBOX_COMPAT_SENTINEL", "inherited")

    backend = LocalSandboxBackend(config=SandboxBackendConfig())
    result = await backend.execute(
        "import os; print(os.environ['SANDBOX_COMPAT_SENTINEL'])",
        str(workspace),
    )

    assert result.success
    assert result.stdout.strip() == "inherited"
    assert result.metadata["isolation"] == "process"
    assert result.metadata["environment_policy"] == "inherit"
    assert result.metadata["network_policy"] == "inherit"


@pytest.mark.asyncio
async def test_execute_does_not_block_the_asyncio_event_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = LocalSandboxBackend(config=SandboxBackendConfig(timeout=5))

    execution = asyncio.create_task(
        backend.execute("import time; time.sleep(0.35)", str(workspace))
    )
    started = time.perf_counter()
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
    assert not execution.done()
    assert (await execution).success


@pytest.mark.asyncio
async def test_bubblewrap_missing_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    backend = _strict_backend()

    with pytest.raises(RuntimeError, match="bwrap.*unavailable"):
        await backend.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            SandboxBackendConfig(
                isolation="bubblewrap",
                environment_policy="inherit",
                network_policy="disabled",
            ),
            "environment_policy='allowlist'",
        ),
        (
            SandboxBackendConfig(
                isolation="bubblewrap",
                environment_policy="allowlist",
                network_policy="inherit",
            ),
            "network_policy='disabled'",
        ),
        (
            SandboxBackendConfig(
                isolation="process",
                environment_policy="inherit",
                network_policy="disabled",
            ),
            "cannot enforce",
        ),
    ],
)
async def test_unenforceable_isolation_combinations_fail_closed(
    config: SandboxBackendConfig,
    message: str,
) -> None:
    backend = LocalSandboxBackend(config=config)

    with pytest.raises(ValueError, match=message):
        await backend.initialize()


@requires_bwrap
@pytest.mark.asyncio
async def test_bubblewrap_exposes_only_frozen_public_symlink_targets(
    tmp_path: Path,
) -> None:
    public = tmp_path / "dataset" / "prepared" / "public"
    private = tmp_path / "dataset" / "prepared" / "private"
    workspace = tmp_path / "run" / "sandbox"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    workspace.mkdir(parents=True)
    public_file = public / "train.csv"
    private_file = private / "answer.csv"
    outside_file = tmp_path / "host-only.txt"
    public_file.write_text("feature\n1\n", encoding="utf-8")
    private_file.write_text("secret-answer", encoding="utf-8")
    outside_file.write_text("host-only", encoding="utf-8")
    (workspace / "train.csv").symlink_to(public_file)

    backend = _strict_backend()
    probe = {
        "private": str(private_file),
        "outside": str(outside_file),
    }
    code = f"""
import json
import os
from pathlib import Path

probe = {probe!r}
result = {{
    "public": Path("train.csv").read_text(),
    "private_exists": Path(probe["private"]).exists(),
    "outside_exists": Path(probe["outside"]).exists(),
}}
for label in ("private", "outside"):
    try:
        Path(probe[label]).read_text()
    except FileNotFoundError:
        result[label + "_read"] = "FileNotFoundError"
Path("solver-output.txt").write_text("written")
try:
    Path(probe["outside"]).write_text("tampered")
except OSError as exc:
    result["outside_write_errno"] = exc.errno
print(json.dumps(result, sort_keys=True))
"""

    result = await backend.execute(code, str(workspace))

    assert result.success, result.stderr
    observed = json.loads(result.stdout)
    assert observed["public"] == "feature\n1\n"
    assert observed["private_exists"] is False
    assert observed["outside_exists"] is False
    assert observed["private_read"] == "FileNotFoundError"
    assert observed["outside_read"] == "FileNotFoundError"
    assert observed["outside_write_errno"] in {2, 30}
    assert (workspace / "solver-output.txt").read_text() == "written"
    assert outside_file.read_text() == "host-only"
    assert result.metadata["isolation"] == "bubblewrap"
    assert result.metadata["network_policy"] == "disabled"
    assert result.metadata["frozen_input_mount_count"] == 1


@requires_bwrap
@pytest.mark.asyncio
async def test_agent_created_symlink_is_never_added_to_frozen_mounts(
    tmp_path: Path,
) -> None:
    public = tmp_path / "dataset" / "public"
    private = tmp_path / "dataset" / "private"
    workspace = tmp_path / "workspace"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    workspace.mkdir()
    (public / "train.csv").write_text("public", encoding="utf-8")
    private_file = private / "answer.csv"
    private_file.write_text("private", encoding="utf-8")
    (workspace / "train.csv").symlink_to(public / "train.csv")

    backend = _strict_backend()
    first = await backend.execute(
        f"""
import os
from pathlib import Path
os.symlink({str(private_file)!r}, "late-private")
try:
    Path("late-private").read_text()
except FileNotFoundError:
    print("FileNotFoundError")
""",
        str(workspace),
    )
    second = await backend.execute(
        """
from pathlib import Path
try:
    Path("late-private").read_text()
except FileNotFoundError:
    print("FileNotFoundError")
""",
        str(workspace),
    )

    assert first.success, first.stderr
    assert first.stdout.strip() == "FileNotFoundError"
    assert second.success, second.stderr
    assert second.stdout.strip() == "FileNotFoundError"
    assert first.metadata["frozen_input_mount_count"] == 1
    assert second.metadata["frozen_input_mount_count"] == 1


@requires_bwrap
@pytest.mark.asyncio
async def test_bubblewrap_environment_drops_credentials_and_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-leak")
    monkeypatch.setenv("UNLISTED_HOST_VALUE", "must-not-leak")
    backend = LocalSandboxBackend(
        config=SandboxBackendConfig(
            isolation="bubblewrap",
            environment_policy="allowlist",
            network_policy="disabled",
            env_vars={"EXPLICIT_SAFE_VALUE": "visible"},
        )
    )

    result = await backend.execute(
        """
import json
import os
print(json.dumps({
    "explicit": os.environ.get("EXPLICIT_SAFE_VALUE"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "api_key": os.environ.get("DEEPSEEK_API_KEY"),
    "proxy": os.environ.get("HTTPS_PROXY"),
    "unlisted": os.environ.get("UNLISTED_HOST_VALUE"),
    "home": os.environ["HOME"],
    "tmp": os.environ["TMPDIR"],
}, sort_keys=True))
""",
        str(workspace),
    )

    assert result.success, result.stderr
    observed = json.loads(result.stdout)
    assert observed["explicit"] == "visible"
    assert observed["openai"] is None
    assert observed["api_key"] is None
    assert observed["proxy"] is None
    assert observed["unlisted"] is None
    assert Path(observed["home"]).is_relative_to(workspace)
    assert Path(observed["tmp"]).is_relative_to(workspace)


@pytest.mark.asyncio
async def test_allowlist_rejects_explicit_credentials() -> None:
    backend = LocalSandboxBackend(
        config=SandboxBackendConfig(
            isolation="process",
            environment_policy="allowlist",
            network_policy="inherit",
            env_vars={"DEEPSEEK_API_KEY": "secret"},
        )
    )

    with pytest.raises(ValueError, match="forbidden"):
        await backend.initialize()


@requires_bwrap
@pytest.mark.asyncio
async def test_timeout_kills_descendant_process_group(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "orphan-marker.txt"
    child_code = (
        "import time; from pathlib import Path; time.sleep(0.8); "
        f"Path({str(marker)!r}).write_text('orphan')"
    )
    parent_code = f"""
import subprocess
import sys
import time
subprocess.Popen([sys.executable, "-c", {child_code!r}])
time.sleep(10)
"""
    backend = _strict_backend()

    result = await backend.execute(parent_code, str(workspace), timeout=0.2)
    await asyncio.sleep(1.0)

    assert not result.success
    assert result.exc_type == "TimeoutError"
    assert not marker.exists()


@requires_bwrap
@pytest.mark.asyncio
async def test_scientific_runtime_remains_importable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = _strict_backend()

    result = await backend.execute(
        "import numpy, pandas; print(numpy.__version__, pandas.__version__)",
        str(workspace),
    )

    assert result.success, result.stderr
    assert len(result.stdout.split()) == 2


@requires_bwrap
@pytest.mark.asyncio
async def test_bubblewrap_cannot_modify_host_scientific_packages(tmp_path: Path) -> None:
    import numpy

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    numpy_init = Path(numpy.__file__).resolve()
    original = numpy_init.read_bytes()
    backend = _strict_backend()

    result = await backend.execute(
        f"""
from pathlib import Path
target = Path({str(numpy_init)!r})
try:
    target.write_text("host environment was modified")
except OSError as exc:
    print(exc.errno)
else:
    raise AssertionError("host package unexpectedly writable")
""",
        str(workspace),
    )

    assert result.success, result.stderr
    assert int(result.stdout.strip()) == 30
    assert numpy_init.read_bytes() == original


@requires_bwrap
@pytest.mark.asyncio
async def test_network_namespace_has_no_external_route(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = _strict_backend()

    result = await backend.execute(
        """
import socket
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("1.1.1.1", 53))
except OSError:
    print("blocked")
else:
    raise AssertionError("network unexpectedly available")
finally:
    s.close()
""",
        str(workspace),
    )

    assert result.success, result.stderr
    assert result.stdout.strip() == "blocked"


def test_config_defaults_preserve_process_and_inherit() -> None:
    config = SandboxBackendConfig()

    assert config.isolation == "process"
    assert config.environment_policy == "inherit"
    assert config.network_policy == "inherit"
