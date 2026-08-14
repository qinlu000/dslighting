"""Composition root for the task-context deep module."""

from __future__ import annotations

from typing import Any

from dslighting.config import DSLightingConfig
from dslighting.core.task_context import TaskContextBuilder
from dslighting.services.data_analysis_provider import create_data_perception_runtime

_UNSET = object()


def create_task_context_builder(
    config: DSLightingConfig,
    *,
    perception_runtime: Any = _UNSET,
) -> TaskContextBuilder:
    """Wire the main data-perception runtime into task-context assembly.

    Passing ``perception_runtime=None`` deliberately disables perception;
    omitting it constructs the runtime from ``config.data_analysis``.
    """

    if perception_runtime is _UNSET:
        perception_runtime = create_data_perception_runtime(config)
    return TaskContextBuilder(perception_runtime)


__all__ = ["create_task_context_builder"]
