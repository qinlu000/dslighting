"""Data-contained process environment for perception-skills experiments."""

from __future__ import annotations

import os
from pathlib import Path

PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


class DataContainedRuntime:
    """Keep experiment state and caches below the repository data directory."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.data_root = self.project_root / "data"

    @property
    def python(self) -> Path:
        return self.project_root / ".venv" / "bin" / "python"

    def environment_for(self, runtime_id: str) -> dict[str, str]:
        if not runtime_id or Path(runtime_id).name != runtime_id or runtime_id in {".", ".."}:
            raise ValueError("runtime ID must be a safe path component")
        root = self.data_root / "runtime" / "perception_skills" / runtime_id
        paths = {
            "HOME": root / "home",
            "TMPDIR": root / "tmp",
            "XDG_CACHE_HOME": root / "xdg" / "cache",
            "XDG_CONFIG_HOME": root / "xdg" / "config",
            "XDG_DATA_HOME": root / "xdg" / "data",
            "XDG_STATE_HOME": root / "xdg" / "state",
            "XDG_RUNTIME_DIR": root / "xdg" / "runtime",
            "PYTHONPYCACHEPREFIX": root / "cache" / "python",
            "PIP_CACHE_DIR": self.data_root / ".uv" / "pip-cache",
            "HF_HOME": self.data_root / "cache" / "huggingface",
            "TORCH_HOME": self.data_root / "cache" / "torch",
            "MPLCONFIGDIR": self.data_root / "cache" / "matplotlib",
            "NUMBA_CACHE_DIR": self.data_root / "cache" / "numba",
            "RUFF_CACHE_DIR": self.data_root / "cache" / "ruff",
            "UV_CACHE_DIR": self.data_root / ".uv" / "cache",
            "UV_INSTALL_DIR": self.data_root / ".uv" / "bin",
            "UV_PYTHON_BIN_DIR": self.data_root / ".uv" / "bin",
            "UV_PYTHON_INSTALL_DIR": self.data_root / ".uv" / "python",
            "UV_TOOL_DIR": self.data_root / ".uv" / "tools",
            "UV_TOOL_BIN_DIR": self.data_root / ".uv" / "bin",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        for name in PROXY_VARIABLES:
            environment.pop(name, None)
        environment.update({name: str(path) for name, path in paths.items()})
        environment.update(
            {
                "TMP": str(paths["TMPDIR"]),
                "TEMP": str(paths["TMPDIR"]),
                "PYTHONUNBUFFERED": "1",
                "UV_OFFLINE": "1",
                "PIP_NO_INDEX": "1",
            }
        )
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
        environment.pop("VIRTUAL_ENV", None)
        return environment

    def activate(self, runtime_id: str) -> None:
        """Apply the contained environment to the dedicated experiment process."""

        environment = self.environment_for(runtime_id)
        os.environ.clear()
        os.environ.update(environment)
