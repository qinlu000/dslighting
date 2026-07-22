"""Frozen DataCOPE discovery protocol used by the DABench experiment."""

from __future__ import annotations

PAPER_DISCOVERY_ROUNDS = 3
PAPER_TRAJECTORIES_PER_TASK = 10
PAPER_REACT_MAX_STEPS = 10
PAPER_EXPLORE_TEMPERATURE = 1.0
PAPER_EVALUATION_TEMPERATURE = 0.0


def validate_round_index(round_index: int) -> int:
    """Return a valid zero-based discovery round index."""

    if not 0 <= round_index < PAPER_DISCOVERY_ROUNDS:
        raise ValueError(
            f"round_index must be in [0, {PAPER_DISCOVERY_ROUNDS - 1}]"
        )
    return round_index


def validate_sample_index(sample_index: int) -> int:
    """Return a valid zero-based trajectory sample index."""

    if not 0 <= sample_index < PAPER_TRAJECTORIES_PER_TASK:
        raise ValueError(
            f"sample_index must be in [0, {PAPER_TRAJECTORIES_PER_TASK - 1}]"
        )
    return sample_index


__all__ = [
    "PAPER_DISCOVERY_ROUNDS",
    "PAPER_EVALUATION_TEMPERATURE",
    "PAPER_EXPLORE_TEMPERATURE",
    "PAPER_REACT_MAX_STEPS",
    "PAPER_TRAJECTORIES_PER_TASK",
    "validate_round_index",
    "validate_sample_index",
]
