"""Frozen experiment profiles for perception-skills ablations.

The public CLI deliberately exposes only experiment choices.  Runtime details
that define the protocol live here and are copied into every batch manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "data_card_ablation"


@dataclass(frozen=True)
class RuntimePolicy:
    workflow: str
    model: str
    task_concurrency: int
    llm_global_concurrency: int
    llm_per_key_concurrency: int
    llm_max_retries: int
    llm_thinking: bool
    sandbox_timeout_seconds: int
    sandbox_backend: str
    sandbox_environment_policy: str
    sandbox_network_policy: str
    device_admission: str
    checkpoint_resume_enabled: bool


@dataclass(frozen=True)
class PerceptionSkillCondition:
    """One frozen annotation condition presented to the downstream agent."""

    perception_skill_id: str
    skill_entrypoints: tuple[Path, ...]
    annotations_dir: Path

    @property
    def is_control(self) -> bool:
        return not self.skill_entrypoints


@dataclass(frozen=True)
class ExperimentSelection:
    tasks: tuple[str, ...] = ()
    tasks_file: Path | None = None
    limit: int | None = None


@dataclass(frozen=True)
class ProtocolOverrides:
    model: str | None = None
    workflow: str | None = None
    task_concurrency: int | None = None

    @property
    def active(self) -> bool:
        return any(
            value is not None for value in (self.model, self.workflow, self.task_concurrency)
        )


@dataclass(frozen=True)
class ExperimentRequest:
    execute: bool
    repetitions: int
    selection: ExperimentSelection
    resume_run_id: str | None = None
    allow_protocol_override: bool = False
    overrides: ProtocolOverrides = ProtocolOverrides()


@dataclass(frozen=True)
class PerceptionSkillsExperimentProfile:
    """One immutable, named experimental protocol."""

    profile_id: str
    schema_version: str
    benchmark: str
    data_root: Path
    annotations_root: Path
    skills_manifest: Path
    dataset_family_manifest: Path | None
    conditions: tuple[str, ...]
    runtime: RuntimePolicy

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "data_root",
            "annotations_root",
            "skills_manifest",
            "dataset_family_manifest",
        ):
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        payload["conditions"] = list(self.conditions)
        return payload

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.as_manifest(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


_COMMON_RUNTIME: Mapping[str, Any] = {
    "workflow": "react",
    "model": "openai/DeepSeek-V4-Flash",
    "llm_max_retries": 10,
    "llm_thinking": False,
    "sandbox_timeout_seconds": 2 * 60 * 60,
    "sandbox_backend": "bubblewrap",
    "sandbox_environment_policy": "allowlist",
    "sandbox_network_policy": "disabled",
    "device_admission": "cpu_default",
    "checkpoint_resume_enabled": True,
}


def _profiles() -> dict[str, PerceptionSkillsExperimentProfile]:
    dabench = PerceptionSkillsExperimentProfile(
        profile_id="perception-skills-dabench-v3",
        schema_version="perception_skills_experiment_profile_v1",
        benchmark="dabench",
        data_root=PROJECT_ROOT / "data" / "releases" / "dabench_perception_clean_v1",
        annotations_root=EXPERIMENT_ROOT / "artifacts" / "dabench_annotations",
        skills_manifest=EXPERIMENT_ROOT / "skills" / "dabench_perception_skills.json",
        dataset_family_manifest=(
            PROJECT_ROOT
            / "data"
            / "releases"
            / "dabench_perception_clean_v1"
            / "dabench_perception_dataset_manifest.json"
        ),
        conditions=("main", "no-added-skill-v1", "dataset-semantics-v1"),
        runtime=RuntimePolicy(
            task_concurrency=257,
            llm_global_concurrency=257,
            llm_per_key_concurrency=257,
            **_COMMON_RUNTIME,
        ),
    )
    moscibench = PerceptionSkillsExperimentProfile(
        profile_id="perception-skills-moscibench-v2",
        schema_version="perception_skills_experiment_profile_v1",
        benchmark="moscibench",
        data_root=PROJECT_ROOT / "data" / "releases" / "moscibench_full" / "moscibench",
        annotations_root=EXPERIMENT_ROOT / "artifacts" / "annotations",
        skills_manifest=EXPERIMENT_ROOT / "skills" / "perception_skills.json",
        dataset_family_manifest=None,
        conditions=(
            "main",
            "no-added-skill-v1",
            "dataset-semantics-v1",
            "scientific-modalities-v1",
        ),
        runtime=RuntimePolicy(
            task_concurrency=20,
            llm_global_concurrency=20,
            llm_per_key_concurrency=20,
            **_COMMON_RUNTIME,
        ),
    )
    return {profile.benchmark: profile for profile in (dabench, moscibench)}


def load_profile(benchmark: str) -> PerceptionSkillsExperimentProfile:
    """Load a built-in profile by benchmark name or full profile ID."""

    normalized = benchmark.strip().lower()
    profiles = _profiles()
    for profile in profiles.values():
        if normalized in {profile.benchmark, profile.profile_id.lower()}:
            return profile
    supported = ", ".join(sorted(profiles))
    raise ValueError(f"unsupported perception-skills benchmark {benchmark!r}; choose {supported}")
