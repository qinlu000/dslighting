"""Launch the DataCOPE bridge in an isolated Python environment."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .protocol import validate_round_index
from .split import DEFAULT_FAMILY_MANIFEST

BRIDGE_PATH = Path(__file__).with_name("datacope_bridge.py")
DEFAULT_SIDECAR_TIMEOUT_SECONDS = 60 * 60
DEFAULT_CODEX_MODEL = "gpt-5.5"
_BRIDGE_FILES = tuple(
    BRIDGE_PATH.with_name(name)
    for name in ("adapter.py", "datacope_bridge.py", "protocol.py", "split.py")
)
_DISCOVERY_ENV_ALLOWLIST = {
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
}


@dataclass(frozen=True)
class CodexRuntime:
    executable: Path
    root: Path
    version: str
    sha256: str


def _python_executable(value: str) -> Path:
    candidate = shutil.which(value) if Path(value).name == value else value
    if not candidate:
        raise ValueError(f"Python executable does not exist: {value}")
    path = Path(os.path.abspath(Path(candidate).expanduser()))
    if not path.is_file():
        raise ValueError(f"Python executable does not exist: {path}")
    return path


def _codex_home_and_auth() -> tuple[Path, Path]:
    codex_home = Path(
        os.environ.get("CODEX_HOME", Path.home() / ".codex")
    ).expanduser().resolve()
    auth_file = codex_home / "auth.json"
    if auth_file.is_symlink() or not auth_file.is_file():
        raise ValueError("Codex ChatGPT login is required; run `codex login` first")
    return codex_home, auth_file


def _codex_runtime(value: str) -> CodexRuntime:
    """Resolve an installed CLI entrypoint to its native runtime."""

    candidate = shutil.which(value) if Path(value).name == value else value
    if not candidate:
        raise ValueError(f"Codex CLI does not exist: {value}")
    entrypoint = Path(os.path.abspath(Path(candidate).expanduser()))
    if not entrypoint.is_file():
        raise ValueError(f"Codex CLI does not exist: {entrypoint}")

    resolved = entrypoint.resolve()
    if resolved.name == "codex.js":
        package_root = resolved.parent.parent
        native_candidates = sorted(
            path.resolve()
            for path in package_root.glob(
                "node_modules/@openai/codex-*/vendor/*/bin/codex"
            )
            if path.is_file()
        )
        if len(native_candidates) != 1:
            raise ValueError(
                "Could not resolve exactly one native Codex binary from the npm install"
            )
        executable = native_candidates[0]
    else:
        executable = resolved

    runtime_root = executable.parent
    for parent in (executable.parent, executable.parent.parent):
        if (parent / "codex-package.json").is_file():
            runtime_root = parent
            break

    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not execute Codex CLI: {executable}") from exc
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or re.fullmatch(r"codex-cli \d+\.\d+\.\d+", version) is None:
        raise ValueError(f"Unexpected Codex CLI version response: {version or 'empty'}")
    with executable.open("rb") as binary:
        sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    return CodexRuntime(
        executable=executable,
        root=runtime_root,
        version=version,
        sha256=sha256,
    )


def _discovery_environment(
    python_executable: Path,
    codex_home: Path,
    codex_runtime_root: Path,
) -> dict[str, str]:
    """Build a small environment without benchmark/API endpoint secrets."""

    environment = {
        name: os.environ[name]
        for name in _DISCOVERY_ENV_ALLOWLIST
        if os.environ.get(name)
    }
    environment.update(
        {
            "CODEX_HOME": str(codex_home),
            "HOME": "/tmp/datacope-home",
            "PATH": ":".join(
                (
                    str(codex_runtime_root / "codex-path"),
                    str(codex_runtime_root / "bin"),
                    str(python_executable.parent),
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                )
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": "/tmp",
        }
    )
    venv_root = python_executable.parent.parent
    if (venv_root / "pyvenv.cfg").is_file():
        environment["VIRTUAL_ENV"] = str(venv_root)
    return environment


def _runtime_mounts(python_executable: Path) -> tuple[Path, ...]:
    """Return the minimum host runtime roots needed by the selected Python."""

    candidates = [
        Path("/usr"),
        Path("/bin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ssl"),
        Path("/etc/pki"),
        Path("/etc/resolv.conf"),
        Path("/etc/hosts"),
        Path("/etc/host.conf"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/gai.conf"),
        Path("/etc/passwd"),
        Path("/etc/group"),
        Path("/etc/localtime"),
        Path("/etc/ca-certificates.conf"),
        python_executable.parent.parent,
        python_executable.resolve().parent.parent,
    ]
    if python_executable.is_symlink():
        link_target = Path(os.readlink(python_executable))
        if not link_target.is_absolute():
            link_target = python_executable.parent / link_target
        candidates.append(link_target.parent.parent)
    resolv_target = Path("/etc/resolv.conf").resolve()
    if resolv_target.exists() and not resolv_target.is_relative_to(Path("/etc")):
        candidates.append(resolv_target)

    mounts: list[Path] = []
    for candidate in candidates:
        candidate = candidate.absolute()
        if candidate == Path("/") or not candidate.exists():
            continue
        if any(candidate == mounted or candidate.is_relative_to(mounted) for mounted in mounts):
            continue
        mounts.append(candidate)
    return tuple(mounts)


def _require_empty_output(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"{label} must be an empty or new directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _discovery_command(
    bridge_command: list[str],
    *,
    python_executable: Path,
    datacope_root: Path,
    predictions_dir: Path,
    verified_dir: Path,
    data_dir: Path,
    skill_dir: Path,
    family_manifest: Path,
    auth_file: Path,
    codex_runtime_root: Path,
    previous_predictions_dirs: Sequence[Path] = (),
    previous_skill_dir: Path | None = None,
    extra_read_only: Sequence[Path] = (),
) -> list[str]:
    """Wrap discovery in a read-allowlisted bubblewrap filesystem."""

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise ValueError("bubblewrap is required for leak-free skill discovery")

    read_only = [
        datacope_root,
        predictions_dir,
        data_dir,
        family_manifest,
        auth_file,
        codex_runtime_root,
        *_BRIDGE_FILES,
        *previous_predictions_dirs,
        *extra_read_only,
    ]
    if previous_skill_dir is not None:
        read_only.append(previous_skill_dir)
    missing = [str(path) for path in read_only if not path.exists()]
    if missing:
        raise ValueError(f"Discovery input does not exist: {missing[0]}")

    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in _runtime_mounts(python_executable):
        command.extend(("--ro-bind", str(path), str(path)))
    for path in read_only:
        command.extend(("--ro-bind", str(path), str(path)))
    for path in (verified_dir, skill_dir):
        command.extend(("--bind", str(path), str(path)))
    command.extend(
        (
            "--remount-ro",
            "/",
            "--chdir",
            str(datacope_root),
            "--",
            *bridge_command,
        )
    )
    return command


def run_sidecar(
    mode: str,
    *,
    datacope_root: Path,
    predictions_dir: Path,
    verified_dir: Path,
    python_executable: str = sys.executable,
    data_dir: Path | None = None,
    skill_dir: Path | None = None,
    family_manifest: Path | None = DEFAULT_FAMILY_MANIFEST,
    model: str | None = DEFAULT_CODEX_MODEL,
    codex_executable: str = "codex",
    round_index: int = 0,
    previous_predictions_dirs: Sequence[Path] = (),
    previous_skill_dir: Path | None = None,
    timeout_seconds: int = DEFAULT_SIDECAR_TIMEOUT_SECONDS,
) -> int:
    """Run verification or skill discovery and return the child exit code."""

    if mode not in {"verify", "discover"}:
        raise ValueError(f"Unsupported sidecar mode: {mode}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    resolved_round = validate_round_index(round_index)

    python_path = _python_executable(python_executable)
    resolved_datacope = Path(datacope_root).expanduser().resolve()
    resolved_predictions = Path(predictions_dir).expanduser().resolve()
    resolved_verified = Path(verified_dir).expanduser().resolve()

    command = [
        str(python_path),
        str(BRIDGE_PATH),
        mode,
        "--datacope-root",
        str(resolved_datacope),
        "--predictions-dir",
        str(resolved_predictions),
        "--verified-dir",
        str(resolved_verified),
    ]
    environment = None
    resolved_previous_predictions: tuple[Path, ...] = ()
    resolved_previous_skill: Path | None = None
    if mode == "verify" and (
        resolved_round != 0 or previous_predictions_dirs or previous_skill_dir is not None
    ):
        raise ValueError("verify checks one initial round; iterative inputs belong to discover")
    if mode == "discover":
        if data_dir is None or skill_dir is None or family_manifest is None or not model:
            raise ValueError("discover requires data_dir, skill_dir, family_manifest, and model")
        resolved_data = Path(data_dir).expanduser().resolve()
        resolved_skill = Path(skill_dir).expanduser().resolve()
        resolved_manifest = Path(family_manifest).expanduser().resolve()
        resolved_previous_predictions = tuple(
            Path(path).expanduser().resolve() for path in previous_predictions_dirs
        )
        if len(resolved_previous_predictions) != resolved_round:
            raise ValueError(
                f"Discovery round {resolved_round} requires {resolved_round} ordered previous "
                "prediction directories"
            )
        if resolved_round == 0:
            if previous_skill_dir is not None:
                raise ValueError("Discovery round 0 must not receive a previous skill")
        else:
            if previous_skill_dir is None:
                raise ValueError(f"Discovery round {resolved_round} requires the previous skill")
            resolved_previous_skill = Path(previous_skill_dir).expanduser().resolve()
            if not (resolved_previous_skill / "SKILL.md").is_file():
                raise ValueError(f"Previous skill is missing SKILL.md: {resolved_previous_skill}")
        codex_runtime = _codex_runtime(codex_executable)
        command.extend(
            [
                "--round-index",
                str(resolved_round),
                "--data-dir",
                str(resolved_data),
                "--skill-dir",
                str(resolved_skill),
                "--family-manifest",
                str(resolved_manifest),
                "--model",
                model,
                "--codex-bin",
                str(codex_runtime.executable),
                "--codex-version",
                codex_runtime.version,
                "--codex-sha256",
                codex_runtime.sha256,
            ]
        )
        for previous_dir in resolved_previous_predictions:
            command.extend(("--previous-predictions-dir", str(previous_dir)))
        if resolved_previous_skill is not None:
            command.extend(("--previous-skill-dir", str(resolved_previous_skill)))
        resolved_verified = _require_empty_output(resolved_verified, "Verified output directory")
        resolved_skill = _require_empty_output(resolved_skill, "Skill output directory")
        if resolved_verified == resolved_skill:
            raise ValueError("Verified and skill output directories must be different")
        codex_home, auth_file = _codex_home_and_auth()
        environment = _discovery_environment(
            python_path,
            codex_home,
            codex_runtime.root,
        )
        command = _discovery_command(
            command,
            python_executable=python_path,
            datacope_root=resolved_datacope,
            predictions_dir=resolved_predictions,
            verified_dir=resolved_verified,
            data_dir=resolved_data,
            skill_dir=resolved_skill,
            family_manifest=resolved_manifest,
            auth_file=auth_file,
            codex_runtime_root=codex_runtime.root,
            previous_predictions_dirs=resolved_previous_predictions,
            previous_skill_dir=resolved_previous_skill,
        )

    try:
        completed = subprocess.run(
            command,
            check=False,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"DataCOPE {mode} timed out after {timeout_seconds} seconds") from exc
    return completed.returncode
