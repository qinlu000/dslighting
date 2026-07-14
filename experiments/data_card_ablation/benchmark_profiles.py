"""Experiment-owned benchmark profiles for data-card ablations.

Core task-context code is benchmark agnostic.  This module contains the small
amount of benchmark knowledge needed to keep each experiment on its reviewed
runtime profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkAblationProfile:
    source_id: str
    default_workflow: str
    l2_guidance_path: Path
    config_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"

_PROFILES = {
    "dabench": BenchmarkAblationProfile(
        source_id="dabench",
        default_workflow="aide",
        l2_guidance_path=_ARTIFACT_ROOT / "dabench_l2_guidance.md",
    ),
    "moscibench": BenchmarkAblationProfile(
        source_id="moscibench",
        default_workflow="react",
        l2_guidance_path=_ARTIFACT_ROOT / "moscibench_l2_guidance.md",
        config_overrides={
            "data_analysis": {
                "cache_enabled": False,
                "profile": "full",
                "max_artifacts": 24,
                "max_report_chars": 80000,
            },
            "agent_runtime": {
                "max_steps": 10,
                "observation": {
                    "max_tokens": 32000,
                    "head_tokens": 16000,
                    "tail_tokens": 16000,
                    "max_chars": 120000,
                },
                "context": {
                    "max_history_chars": 240000,
                    "keep_recent_turns": 20,
                    "recent_observation_window": 12,
                    "summary_trigger_turns": 24,
                    "summary_max_chars": 12000,
                },
            },
            "output_contract": {
                "require_output_before_completion": True,
                "missing_output_feedback_retries": 2,
            },
        },
    ),
}


def get_ablation_profile(source_id: str) -> BenchmarkAblationProfile:
    """Return the reviewed profile for a supported benchmark source."""

    try:
        return _PROFILES[source_id]
    except KeyError as error:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Benchmark source {source_id!r} has no reviewed data-card ablation "
            f"profile; supported sources: {supported}."
        ) from error


__all__ = ["BenchmarkAblationProfile", "get_ablation_profile"]
