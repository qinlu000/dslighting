"""Experiment constants shared by the MoSciBench adapter."""

PAPER_TRAJECTORIES_PER_TASK = 3
PAPER_DISCOVERY_ROUNDS = 3
EXPLORE_TEMPERATURE = 1.0
TEST_TEMPERATURE = 0.0


def validate_round_index(value: int) -> int:
    value = int(value)
    if not 0 <= value < PAPER_DISCOVERY_ROUNDS:
        raise ValueError(f"round_index must be in [0, {PAPER_DISCOVERY_ROUNDS - 1}]")
    return value


def validate_sample_index(value: int) -> int:
    value = int(value)
    if not 0 <= value < PAPER_TRAJECTORIES_PER_TASK:
        raise ValueError(f"sample_index must be in [0, {PAPER_TRAJECTORIES_PER_TASK - 1}]")
    return value
