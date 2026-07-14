"""Shared artifact-reading primitives for task-context inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ArtifactNotFoundError, ArtifactProvenance, ArtifactReadError


def read_utf8_artifact(
    path: Path,
    *,
    artifact_label: str,
) -> tuple[str, ArtifactProvenance]:
    """Read one required artifact as UTF-8 and capture its exact digest."""

    if not path.is_file():
        raise ArtifactNotFoundError(f"Selected {artifact_label} artifact not found: {path}")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ArtifactReadError(f"Could not read {artifact_label} artifact: {path}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactReadError(f"{artifact_label} artifact is not valid UTF-8: {path}") from error
    return text, ArtifactProvenance(path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest())


def normalize_markdown(text: str) -> str:
    """Normalize line endings and surrounding whitespace deterministically."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


__all__ = ["normalize_markdown", "read_utf8_artifact"]
