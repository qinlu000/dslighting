"""Shared AgenticDataBench DataCOPE protocol constants."""

TRAJECTORIES_PER_TASK = 3
DISCOVERY_ROUNDS = 3
EXPLORE_TEMPERATURE = 1.0
TEST_TEMPERATURE = 0.0


def validate_round_index(value: int) -> int:
    value = int(value)
    if not 0 <= value < DISCOVERY_ROUNDS:
        raise ValueError(f"round_index must be in [0, {DISCOVERY_ROUNDS - 1}]")
    return value


def validate_sample_index(value: int) -> int:
    value = int(value)
    if not 0 <= value < TRAJECTORIES_PER_TASK:
        raise ValueError(f"sample_index must be in [0, {TRAJECTORIES_PER_TASK - 1}]")
    return value
