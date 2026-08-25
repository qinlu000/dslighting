"""Local sandbox backend with opt-in bubblewrap isolation."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dslighting.services.sandbox_backends.backends.base import (
    SandboxBackend,
    SandboxBackendConfig,
)
from dslighting.utils.typing import ExecutionResult

logger = logging.getLogger(__name__)


_SAFE_INHERITED_ENV = frozenset(
    {
        "BLIS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TERM",
        "TZ",
        "VECLIB_MAXIMUM_THREADS",
    }
)
_RESERVED_ENV = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "HOME",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SHELLOPTS",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
)
_CREDENTIAL_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTHORIZATION|COOKIE|CREDENTIALS?|PASSWORD|PASSWD|"
    r"PRIVATE_KEY|ACCESS_KEY|SECRET|SESSION_KEY|TOKEN)(?:_|$)"
)


def _is_sensitive_env_name(name: str) -> bool:
    normalized = name.upper()
    if (
        "OPENAI" in normalized
        or "API_KEY" in normalized
        or "APIKEY" in normalized
        or normalized.endswith("_PROXY")
    ):
        return True
    if normalized in {"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}:
        return True
    return _CREDENTIAL_NAME_PATTERN.search(normalized) is not None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


class LocalSandboxBackend(SandboxBackend):
    """Execute Python locally, optionally inside a minimal bubblewrap root.

    ``process`` plus ``inherit`` is the compatibility default.  The opt-in
    ``bubblewrap`` mode is deliberately strict: it requires an allowlisted
    environment and a disabled network, and never falls back to a plain
    process when isolation cannot be established.
    """

    def __init__(
        self,
        config: Optional[SandboxBackendConfig] = None,
        workspace_path: Optional[str] = None,
        env_overrides: Optional[dict] = None,
        auto_matplotlib: bool = False,
    ):
        super().__init__(config)
        self.workspace_path = workspace_path
        self.env_overrides = {
            str(key): str(value)
            for key, value in (env_overrides or {}).items()
            if value is not None
        }
        self.auto_matplotlib = auto_matplotlib
        self._initialized = False
        self._bwrap_path: Optional[str] = None
        self._bubblewrap_workspace: Optional[Path] = None
        self._bubblewrap_input_mounts: Optional[tuple[tuple[Path, Path], ...]] = None
        self._python_executable = Path(sys.executable).absolute()
        self._python_mounts: tuple[Path, ...] = ()

    async def initialize(self) -> None:
        """Validate the selected isolation mode before accepting executions."""
        isolation = self.config.isolation
        environment_policy = self.config.environment_policy
        network_policy = self.config.network_policy

        if isolation not in {"process", "bubblewrap"}:
            raise ValueError(f"Unsupported local sandbox isolation mode: {isolation!r}")
        if environment_policy not in {"inherit", "allowlist"}:
            raise ValueError(f"Unsupported sandbox environment policy: {environment_policy!r}")
        if network_policy not in {"inherit", "disabled"}:
            raise ValueError(f"Unsupported sandbox network policy: {network_policy!r}")
        if environment_policy == "allowlist":
            self._validate_configured_env()

        configured_python = self.config.python_executable or sys.executable
        self._python_executable = Path(configured_python).expanduser().absolute()
        if not self._python_executable.is_file():
            raise ValueError(
                f"sandbox python_executable does not exist: {self._python_executable}"
            )
        python_mounts = [
            self._python_executable.parent.parent,
            self._python_executable.resolve().parent.parent,
        ]
        if self._python_executable.is_symlink():
            target = Path(os.readlink(self._python_executable))
            if not target.is_absolute():
                target = self._python_executable.parent / target
            python_mounts.append(Path(os.path.abspath(target)).parent.parent)
        self._python_mounts = tuple(python_mounts)

        if isolation == "process":
            if network_policy != "inherit":
                raise ValueError(
                    "Local process isolation cannot enforce network_policy='disabled'; "
                    "select isolation='bubblewrap'"
                )
        else:
            if os.name != "posix":
                raise RuntimeError("Bubblewrap isolation is only supported on POSIX systems")
            if environment_policy != "allowlist":
                raise ValueError("Bubblewrap isolation requires environment_policy='allowlist'")
            if network_policy != "disabled":
                raise ValueError("Bubblewrap isolation requires network_policy='disabled'")
            self._bwrap_path = shutil.which("bwrap")
            if self._bwrap_path is None:
                raise RuntimeError(
                    "Bubblewrap isolation was requested, but the 'bwrap' executable is unavailable"
                )

        logger.info(
            "Initializing LocalSandboxBackend "
            "(isolation=%s, env=%s, network=%s, python=%s)",
            isolation,
            environment_policy,
            network_policy,
            self._python_executable,
        )
        self._initialized = True

    async def execute(
        self,
        code: str,
        workspace_path: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code without blocking the caller's asyncio event loop."""
        if not self._initialized:
            await self.initialize()

        effective_timeout = self.config.timeout if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError("Sandbox timeout must be greater than zero")

        effective_workspace_value = workspace_path or self.workspace_path
        if not effective_workspace_value:
            raise ValueError("workspace_path must be provided")
        effective_workspace = Path(effective_workspace_value).resolve()
        effective_workspace.mkdir(parents=True, exist_ok=True)
        if not effective_workspace.is_dir():
            raise ValueError("workspace_path must identify a directory")

        if self.config.isolation == "bubblewrap":
            self._freeze_bubblewrap_inputs(effective_workspace)

        return await self._execute_async(code, effective_workspace, effective_timeout)

    def _freeze_bubblewrap_inputs(self, workspace: Path) -> None:
        """Capture trusted, direct-child workspace symlinks exactly once.

        WorkspaceService creates these links before agent code runs.  Freezing
        this mount list prevents code from creating a later symlink to a private
        host path and having a subsequent execution expose that path.
        """
        if self._bubblewrap_input_mounts is not None:
            if workspace != self._bubblewrap_workspace:
                raise ValueError(
                    "A bubblewrap backend instance cannot switch workspaces after its "
                    "input mount allowlist has been frozen"
                )
            return

        mounts: list[tuple[Path, Path]] = []
        for child in sorted(workspace.iterdir(), key=lambda path: path.name):
            if not child.is_symlink():
                continue

            raw_target = Path(os.readlink(child))
            if raw_target.is_absolute():
                visible_target = Path(os.path.abspath(raw_target))
            else:
                visible_target = Path(os.path.abspath(workspace / raw_target))

            try:
                source_target = child.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise ValueError(f"Workspace input symlink is dangling or cyclic: {child}") from exc

            if _is_relative_to(visible_target, workspace):
                continue
            if source_target == Path("/") or _is_relative_to(workspace, source_target):
                raise ValueError(f"Refusing overly broad workspace input symlink target: {child}")
            if not (source_target.is_file() or source_target.is_dir()):
                raise ValueError(
                    f"Workspace input symlink target is not a regular file or directory: {child}"
                )
            mounts.append((source_target, visible_target))

        self._bubblewrap_workspace = workspace
        self._bubblewrap_input_mounts = tuple(mounts)

    async def _execute_async(
        self,
        code: str,
        workspace_path: Path,
        timeout: float,
    ) -> ExecutionResult:
        if self.auto_matplotlib:
            fixed_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
            logger.debug("Auto-injected matplotlib non-interactive backend")
        else:
            fixed_code = code

        script_name = f"_sandbox_script_{uuid.uuid4().hex}.py"
        script_path = workspace_path / "run" / script_name
        execution_id = uuid.uuid4().hex
        started_at = datetime.utcnow()
        perf_start = time.perf_counter()
        script_path.parent.mkdir(parents=True, exist_ok=True)

        execution_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            exc_type=None,
        )

        try:
            script_path.write_text(fixed_code, encoding="utf-8")
            child_env = self._build_child_env(workspace_path)
            command = self._build_command(script_path, workspace_path, child_env)
            logger.info(
                "Executing script '%s' in local sandbox (timeout: %ss, isolation=%s)",
                script_name,
                timeout,
                self.config.isolation,
            )

            returncode, stdout, stderr, timed_out = await self._run_command(
                command,
                workspace_path,
                child_env,
                timeout,
            )
            if timed_out:
                logger.warning("Script execution timed out; its process group was terminated")
                execution_result = ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr or f"TimeoutError: Execution exceeded {timeout} seconds.",
                    exc_type="TimeoutError",
                )
            else:
                success = returncode == 0
                exc_type = self._extract_exception_type(stderr) if not success else None
                execution_result = ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    exc_type=exc_type,
                )
                status = "succeeded" if success else f"failed (exit code {returncode})"
                logger.info("Script execution finished: %s.", status)
                if not success:
                    logger.error(
                        "Sandbox script failed (exit=%s, exception=%s): %s",
                        returncode,
                        exc_type,
                        stderr,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "An unexpected error occurred during sandbox setup or execution: %s",
                exc,
                exc_info=True,
            )
            execution_result = ExecutionResult(
                success=False,
                stderr=str(exc),
                exc_type=exc.__class__.__name__,
            )
        finally:
            ended_at = datetime.utcnow()
            duration = round(time.perf_counter() - perf_start, 4)
            copied_script_path = self._copy_script_artifact(
                script_path,
                workspace_path,
                script_name,
            )
            execution_result.metadata = {
                "execution_id": execution_id,
                "script_filename": script_name,
                "original_script_path": str(script_path) if script_path.exists() else None,
                "copied_script_path": (str(copied_script_path) if copied_script_path else None),
                "sandbox_cwd": str(workspace_path),
                "started_at_utc": started_at.isoformat() + "Z",
                "ended_at_utc": ended_at.isoformat() + "Z",
                "duration_seconds": duration,
                "backend": "local",
                "isolation": self.config.isolation,
                "environment_policy": self.config.environment_policy,
                "network_policy": self.config.network_policy,
                "frozen_input_mount_count": len(self._bubblewrap_input_mounts or ()),
                "python_executable": str(self._python_executable),
            }

        return execution_result

    def _build_child_env(self, workspace: Path) -> dict[str, str]:
        configured_env = self._configured_env()
        if self.config.environment_policy == "inherit":
            return {**os.environ, **configured_env}

        self._validate_configured_env()

        sandbox_home = workspace / ".sandbox_home"
        sandbox_tmp = workspace / ".sandbox_tmp"
        matplotlib_config = sandbox_home / ".config" / "matplotlib"
        sandbox_tmp.mkdir(parents=True, exist_ok=True)
        matplotlib_config.mkdir(parents=True, exist_ok=True)

        child_env = {key: os.environ[key] for key in _SAFE_INHERITED_ENV if key in os.environ}
        child_env.update(
            {
                "HOME": str(sandbox_home),
                "MPLCONFIGDIR": str(matplotlib_config),
                "PATH": f"{self._python_executable.parent}:/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TEMP": str(sandbox_tmp),
                "TMP": str(sandbox_tmp),
                "TMPDIR": str(sandbox_tmp),
            }
        )
        child_env.update(configured_env)
        return child_env

    def _configured_env(self) -> dict[str, str]:
        return {
            **{
                str(key): str(value)
                for key, value in self.config.env_vars.items()
                if value is not None
            },
            **self.env_overrides,
        }

    def _validate_configured_env(self) -> None:
        configured_env = self._configured_env()
        for name in configured_env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid sandbox environment variable name: {name!r}")
            normalized = name.upper()
            if normalized in _RESERVED_ENV or _is_sensitive_env_name(normalized):
                raise ValueError(f"Environment variable {name!r} is forbidden in allowlist mode")

    def _build_command(
        self,
        script_path: Path,
        workspace: Path,
        child_env: dict[str, str],
    ) -> list[str]:
        if self.config.isolation == "process":
            return [str(self._python_executable), str(script_path)]

        if self._bwrap_path is None or self._bubblewrap_input_mounts is None:
            raise RuntimeError("Bubblewrap backend was not initialized safely")

        command = [
            self._bwrap_path,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--tmpfs",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]

        for source in self._runtime_mounts(workspace):
            command.extend(("--ro-bind", str(source), str(source)))
        for source, destination in self._bubblewrap_input_mounts:
            command.extend(("--ro-bind", str(source), str(destination)))

        command.extend(("--bind", str(workspace), str(workspace)))
        command.extend(("--remount-ro", "/"))
        command.extend(("--chdir", str(workspace)))
        for name, value in sorted(child_env.items()):
            command.extend(("--setenv", name, value))
        command.extend((str(self._python_executable), str(script_path)))
        return command

    def _runtime_mounts(self, workspace: Path) -> tuple[Path, ...]:
        candidates = [
            Path("/usr"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
            # Scientific libraries need these small, non-secret host configs
            # for deterministic font discovery and local-time parsing. Keep the
            # rest of /etc unavailable to sandboxed code.
            Path("/etc/fonts"),
            Path("/etc/localtime"),
            Path("/etc/timezone"),
            *self._python_mounts,
        ]
        mounts: list[Path] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            candidate = candidate.absolute()
            if _is_relative_to(candidate, workspace):
                raise ValueError(
                    "The Python runtime cannot live inside the writable sandbox workspace"
                )
            if any(_is_relative_to(candidate, existing) for existing in mounts):
                continue
            mounts.append(candidate)
        return tuple(mounts)

    async def _run_command(
        self,
        command: list[str],
        workspace: Path,
        child_env: dict[str, str],
        timeout: float,
    ) -> tuple[int, str, str, bool]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            env=child_env,
            start_new_session=True,
        )
        communicate_task = asyncio.create_task(process.communicate())
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=timeout,
            )
            return (
                int(process.returncode or 0),
                self._decode_output(stdout_bytes),
                self._decode_output(stderr_bytes),
                False,
            )
        except asyncio.TimeoutError:
            stdout_bytes, stderr_bytes = await self._terminate_and_collect(
                process,
                communicate_task,
            )
            return (
                int(process.returncode or -signal.SIGKILL),
                self._decode_output(stdout_bytes),
                self._decode_output(stderr_bytes),
                True,
            )
        except BaseException:
            await self._terminate_and_collect(process, communicate_task)
            raise

    @staticmethod
    async def _terminate_and_collect(
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task,
    ) -> tuple[bytes, bytes]:
        def signal_process_group(sig: signal.Signals) -> None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                elif sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except ProcessLookupError:
                pass

        signal_process_group(signal.SIGTERM)
        try:
            return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=1.0)
        except asyncio.TimeoutError:
            signal_process_group(signal.SIGKILL)
            return await asyncio.shield(communicate_task)

    @staticmethod
    def _decode_output(value: bytes | None) -> str:
        return (value or b"").decode("utf-8", errors="replace")

    @staticmethod
    def _extract_exception_type(stderr: str) -> Optional[str]:
        stderr_lines = stderr.strip().split("\n")
        if not stderr_lines:
            return None
        match = re.search(r"^(\w+(?:Error|Exception)):", stderr_lines[-1])
        return match.group(1) if match else None

    @staticmethod
    def _copy_script_artifact(
        script_path: Path,
        workspace_path: Path,
        script_name: str,
    ) -> Optional[Path]:
        artifacts_dir = workspace_path / "artifacts" / "sandbox_scripts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not script_path.exists():
            return None
        copied_script_path = artifacts_dir / script_name
        try:
            shutil.copy2(script_path, copied_script_path)
        except Exception as copy_error:
            logger.error(
                "Failed to copy sandbox script '%s' to artifacts: %s",
                script_name,
                copy_error,
                exc_info=True,
            )
            return None
        return copied_script_path

    async def shutdown(self) -> None:
        """Mark this backend unavailable until it is initialized again."""
        logger.info("Shutting down LocalSandboxBackend")
        self._initialized = False
