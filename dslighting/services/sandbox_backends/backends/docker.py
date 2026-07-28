"""Docker sandbox backend with one persistent container per task workspace."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from dslighting.services.sandbox_backends.backends.base import (
    SandboxBackend,
    SandboxBackendConfig,
)
from dslighting.utils.typing import ExecutionResult

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTHORIZATION|COOKIE|CREDENTIALS?|PASSWORD|PASSWD|"
    r"PRIVATE_KEY|ACCESS_KEY|SECRET|SESSION_KEY|TOKEN)(?:_|$)"
)
_PROTECTED_MOUNT_ROOTS = (
    Path("/bin"),
    Path("/boot"),
    Path("/dev"),
    Path("/etc"),
    Path("/lib"),
    Path("/lib64"),
    Path("/proc"),
    Path("/root"),
    Path("/run"),
    Path("/sbin"),
    Path("/sys"),
    Path("/usr"),
    Path("/var"),
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_docker_input_mounts(
    workspace: Path,
) -> tuple[tuple[Path, Path], ...]:
    """Resolve trusted top-level workspace symlinks into read-only Docker mounts.

    The returned pairs are ``(host_source, container_visible_target)``.  Mounting
    the resolved host source at the symlink's original absolute target keeps
    input links valid inside a container without exposing arbitrary host paths.
    """

    workspace = Path(workspace).expanduser().resolve()
    mounts: list[tuple[Path, Path]] = []
    for child in sorted(workspace.iterdir(), key=lambda path: path.name):
        if not child.is_symlink():
            continue
        raw_target = Path(os.readlink(child))
        visible_target = Path(
            os.path.abspath(
                raw_target if raw_target.is_absolute() else workspace / raw_target
            )
        )
        try:
            source_target = child.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise ValueError(
                f"Workspace input symlink is dangling or cyclic: {child}"
            ) from exc
        if _is_relative_to(visible_target, workspace):
            continue
        if source_target == Path("/") or _is_relative_to(workspace, source_target):
            raise ValueError(f"Refusing overly broad Docker input symlink target: {child}")
        if any(
            visible_target == protected
            or _is_relative_to(visible_target, protected)
            for protected in _PROTECTED_MOUNT_ROOTS
        ):
            raise ValueError(
                f"Refusing Docker input mount over protected container path: {visible_target}"
            )
        if not (source_target.is_file() or source_target.is_dir()):
            raise ValueError(
                "Workspace input symlink target is not a regular file or directory: "
                f"{child}"
            )
        mounts.append((source_target, visible_target))
    return tuple(mounts)


def _default_client_factory() -> Any:
    try:
        import docker
    except ImportError as exc:
        raise ImportError(
            "Docker sandbox support is not installed. Install it with: "
            "pip install 'dslighting[docker]'"
        ) from exc
    return docker.from_env()


class DockerSandboxBackend(SandboxBackend):
    """Execute generated Python in a long-lived, task-scoped Docker container.

    The DSLighting workspace is mounted read-write at ``container_workspace``.
    Trusted direct-child input symlinks created before the first execution are
    resolved once and mounted read-only at their existing absolute targets.
    Symlinks created later by agent code never expand the mount allowlist.
    """

    def __init__(
        self,
        *,
        image: str,
        config: Optional[SandboxBackendConfig] = None,
        container_workspace: str = "/workspace",
        user: str | None = None,
        pids_limit: int = 256,
        python_executable: str = "python",
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.image = str(image or "").strip()
        self.container_workspace = str(container_workspace or "").strip()
        self.user = str(user).strip() if user else None
        self.pids_limit = int(pids_limit)
        self.python_executable = str(python_executable or "python").strip()
        self._client_factory = client_factory or _default_client_factory
        self._client: Any = None
        self._container: Any = None
        self._workspace_host: Path | None = None
        self._input_mounts: tuple[tuple[Path, Path], ...] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Connect to Docker and fail closed on invalid isolation settings."""

        if not self.image:
            raise ValueError("Docker sandbox requires a non-empty image")
        container_workspace = Path(self.container_workspace)
        if not container_workspace.is_absolute() or container_workspace == Path("/"):
            raise ValueError("Docker container_workspace must be an absolute non-root path")
        if self.config.environment_policy not in {"inherit", "allowlist"}:
            raise ValueError(
                f"Unsupported sandbox environment policy: {self.config.environment_policy!r}"
            )
        if self.config.network_policy not in {"inherit", "disabled"}:
            raise ValueError(f"Unsupported sandbox network policy: {self.config.network_policy!r}")
        if self.config.memory_mb <= 0 or self.config.cpu_cores <= 0:
            raise ValueError("Docker memory_mb and cpu_cores must be positive")
        if self.pids_limit <= 0:
            raise ValueError("Docker pids_limit must be positive")
        self._validate_env()

        try:
            self._client = await asyncio.to_thread(self._client_factory)
            await asyncio.to_thread(self._client.ping)
            await asyncio.to_thread(self._client.images.get, self.image)
        except ImportError:
            raise
        except Exception as exc:
            self._client = None
            raise RuntimeError(
                f"Docker is unavailable or image {self.image!r} is not built: {exc}"
            ) from exc

        logger.info(
            "Initializing DockerSandboxBackend (image=%s, network=%s)",
            self.image,
            self.config.network_policy,
        )
        self._initialized = True

    async def execute(
        self,
        code: str,
        workspace_path: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute one Python script while preserving task filesystem state."""

        if not self._initialized:
            await self.initialize()

        effective_timeout = float(self.config.timeout if timeout is None else timeout)
        if effective_timeout <= 0:
            raise ValueError("Sandbox timeout must be greater than zero")
        workspace = Path(workspace_path).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if not workspace.is_dir():
            raise ValueError("workspace_path must identify a directory")

        await self._ensure_container(workspace)
        return await self._execute_script(code, workspace, effective_timeout)

    async def _ensure_container(self, workspace: Path) -> None:
        if self._container is not None:
            if workspace != self._workspace_host:
                raise ValueError(
                    "A Docker sandbox backend instance cannot switch task workspaces"
                )
            return
        if self._client is None:
            raise RuntimeError("Docker backend is not initialized")

        self._workspace_host = workspace
        self._input_mounts = self._freeze_input_mounts(workspace)
        (workspace / "run").mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts" / "sandbox_scripts").mkdir(parents=True, exist_ok=True)
        (workspace / ".sandbox_home" / ".config" / "matplotlib").mkdir(
            parents=True,
            exist_ok=True,
        )
        (workspace / ".sandbox_tmp").mkdir(parents=True, exist_ok=True)

        volumes: dict[str, dict[str, str]] = {
            str(workspace): {"bind": self.container_workspace, "mode": "rw"}
        }
        for source, destination in self._input_mounts:
            source_key = str(source)
            if source_key in volumes and volumes[source_key]["bind"] != str(destination):
                raise ValueError(
                    f"One Docker input source cannot be mounted at multiple targets: {source}"
                )
            volumes[source_key] = {"bind": str(destination), "mode": "ro"}

        run_kwargs: dict[str, Any] = {
            "image": self.image,
            "command": ["sh", "-c", "while :; do sleep 3600; done"],
            "detach": True,
            "working_dir": self.container_workspace,
            "volumes": volumes,
            "environment": self._container_env(),
            "mem_limit": f"{int(self.config.memory_mb)}m",
            "nano_cpus": int(float(self.config.cpu_cores) * 1_000_000_000),
            "pids_limit": self.pids_limit,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "labels": {"dslighting.sandbox": "true", "dslighting.image": self.image},
        }
        if self.config.network_policy == "disabled":
            run_kwargs["network_mode"] = "none"
        if self.user:
            run_kwargs["user"] = self.user

        try:
            self._container = await asyncio.to_thread(self._client.containers.run, **run_kwargs)
        except Exception as exc:
            self._workspace_host = None
            self._input_mounts = None
            raise RuntimeError(f"Failed to start Docker sandbox from {self.image!r}: {exc}") from exc

    async def _execute_script(
        self,
        code: str,
        workspace: Path,
        timeout: float,
    ) -> ExecutionResult:
        script_name = f"_sandbox_script_{uuid.uuid4().hex}.py"
        host_script = workspace / "run" / script_name
        container_script = str(Path(self.container_workspace) / "run" / script_name)
        host_script.write_text(code, encoding="utf-8")
        execution_id = uuid.uuid4().hex
        started_at = datetime.utcnow()
        perf_start = time.perf_counter()
        result = ExecutionResult(success=False, stdout="", stderr="", exc_type=None)

        try:
            exec_task = asyncio.create_task(
                asyncio.to_thread(
                    self._container.exec_run,
                    [self.python_executable, container_script],
                    workdir=self.container_workspace,
                    environment=self._container_env(),
                    demux=True,
                )
            )
            try:
                docker_result = await asyncio.wait_for(exec_task, timeout=timeout)
            except asyncio.TimeoutError:
                await self._remove_container()
                result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"TimeoutError: Execution exceeded {timeout} seconds.",
                    exc_type="TimeoutError",
                )
            else:
                stdout, stderr = self._decode_exec_output(docker_result.output)
                success = int(docker_result.exit_code) == 0
                result = ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exc_type=None if success else self._extract_exception_type(stderr),
                )
        except asyncio.CancelledError:
            await self._remove_container()
            raise
        except Exception as exc:
            logger.error("Docker sandbox execution failed: %s", exc, exc_info=True)
            result = ExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exc_type=exc.__class__.__name__,
            )
        finally:
            artifact = workspace / "artifacts" / "sandbox_scripts" / script_name
            try:
                shutil.copy2(host_script, artifact)
            except OSError as exc:
                logger.warning("Failed to preserve Docker sandbox script %s: %s", script_name, exc)
                artifact = None
            ended_at = datetime.utcnow()
            result.metadata = {
                "execution_id": execution_id,
                "script_filename": script_name,
                "original_script_path": str(host_script),
                "copied_script_path": str(artifact) if artifact else None,
                "sandbox_cwd": self.container_workspace,
                "started_at_utc": started_at.isoformat() + "Z",
                "ended_at_utc": ended_at.isoformat() + "Z",
                "duration_seconds": round(time.perf_counter() - perf_start, 4),
                "backend": "docker",
                "image": self.image,
                "container_id": getattr(self._container, "id", None),
                "environment_policy": self.config.environment_policy,
                "network_policy": self.config.network_policy,
                "frozen_input_mount_count": len(self._input_mounts or ()),
            }
        return result

    def _freeze_input_mounts(self, workspace: Path) -> tuple[tuple[Path, Path], ...]:
        return resolve_docker_input_mounts(workspace)

    def _container_env(self) -> dict[str, str]:
        workspace = Path(self.container_workspace)
        env = {
            "HOME": str(workspace / ".sandbox_home"),
            "MPLCONFIGDIR": str(workspace / ".sandbox_home" / ".config" / "matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TEMP": str(workspace / ".sandbox_tmp"),
            "TMP": str(workspace / ".sandbox_tmp"),
            "TMPDIR": str(workspace / ".sandbox_tmp"),
        }
        env.update({str(key): str(value) for key, value in self.config.env_vars.items()})
        return env

    def _validate_env(self) -> None:
        for name in self.config.env_vars:
            if not _ENV_NAME.fullmatch(str(name)):
                raise ValueError(f"Invalid sandbox environment variable name: {name!r}")
            normalized = str(name).upper()
            if (
                _SENSITIVE_ENV_NAME.search(normalized)
                or "OPENAI" in normalized
                or normalized.endswith("_PROXY")
            ):
                raise ValueError(f"Environment variable {name!r} is forbidden in Docker sandbox")

    @staticmethod
    def _decode_exec_output(output: Any) -> tuple[str, str]:
        if isinstance(output, tuple):
            stdout_raw, stderr_raw = output
        else:
            stdout_raw, stderr_raw = output, b""

        def decode(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        return decode(stdout_raw), decode(stderr_raw)

    @staticmethod
    def _extract_exception_type(stderr: str) -> str | None:
        matches = re.findall(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", stderr, re.MULTILINE)
        return matches[-1] if matches else None

    async def _remove_container(self) -> None:
        container, self._container = self._container, None
        if container is None:
            return
        try:
            await asyncio.to_thread(container.remove, force=True)
        except Exception as exc:
            logger.warning("Failed to remove Docker sandbox container: %s", exc)

    async def shutdown(self) -> None:
        """Remove the task container and close the Docker client."""

        await self._remove_container()
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            try:
                await asyncio.to_thread(client.close)
            except Exception as exc:
                logger.warning("Failed to close Docker client: %s", exc)
        self._workspace_host = None
        self._input_mounts = None
        self._initialized = False
