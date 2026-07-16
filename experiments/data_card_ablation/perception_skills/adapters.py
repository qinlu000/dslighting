"""Benchmark-specific frozen-input validation for perception-skills runs."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from .profile import PerceptionSkillsExperimentProfile


class AdapterPreflightError(RuntimeError):
    """Raised when a frozen benchmark input is absent or inconsistent."""


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise AdapterPreflightError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise AdapterPreflightError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise AdapterPreflightError(f"frozen input tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class BenchmarkAdapter(ABC):
    def __init__(self, profile: PerceptionSkillsExperimentProfile) -> None:
        self.profile = profile

    @abstractmethod
    def preflight(self) -> Mapping[str, Any]:
        """Verify immutable benchmark and annotation inputs."""


class MoSciBenchAdapter(BenchmarkAdapter):
    def preflight(self) -> Mapping[str, Any]:
        data_root = require_directory(self.profile.data_root, "MoSciBench release")
        skills_manifest = require_file(
            self.profile.skills_manifest,
            "MoSciBench perception-skills manifest",
        )
        annotations_root = require_directory(
            self.profile.annotations_root,
            "MoSciBench annotations",
        )
        try:
            manifest = json.loads(skills_manifest.read_text(encoding="utf-8"))
            declared = [item["perception_skill_id"] for item in manifest["perception_skills"]]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise AdapterPreflightError(
                f"invalid MoSciBench perception-skills manifest: {error}"
            ) from error
        expected = [condition for condition in self.profile.conditions if condition != "main"]
        if declared != expected:
            raise AdapterPreflightError(
                f"MoSciBench profile conditions {expected!r} do not match manifest {declared!r}"
            )
        for condition in expected:
            require_directory(annotations_root / condition, f"annotation condition {condition}")
        return {
            "benchmark": self.profile.benchmark,
            "data_root": str(data_root),
            "annotation_conditions": declared,
            "annotation_tree_verified": True,
            "annotation_tree_sha256": tree_sha256(annotations_root),
        }


def adapter_for(profile: PerceptionSkillsExperimentProfile) -> BenchmarkAdapter:
    if profile.benchmark == "dabench":
        from .dabench_adapter import DABenchAdapter

        return DABenchAdapter(profile)
    if profile.benchmark == "moscibench":
        return MoSciBenchAdapter(profile)
    raise ValueError(f"no adapter for benchmark {profile.benchmark!r}")
