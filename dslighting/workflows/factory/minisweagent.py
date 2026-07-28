"""Factory for the optional mini-swe-agent workflow."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from dslighting.benchmark.core.base import BaseBenchmark
from dslighting.error import ConfigurationError
from dslighting.services.workspace import WorkspaceService
from dslighting.workflows.factory.base import BaseWorkflowFactory
from dslighting.workflows.search.minisweagent import (
    MiniSWEAgentSettings,
    MiniSWEAgentTelemetry,
    MiniSWEAgentWorkflow,
)


class MiniSWEAgentWorkflowFactory(BaseWorkflowFactory):
    """Create a mini-swe-agent workflow using DSLighting runtime services."""

    def _get_workflow_name(self) -> str:
        return "mini_swe_agent"

    def create_workflow(
        self,
        config: Any,
        benchmark: Optional[BaseBenchmark] = None,
    ) -> MiniSWEAgentWorkflow:
        raw_params = dict(getattr(config.workflow, "params", None) or {})
        try:
            settings = MiniSWEAgentSettings.model_validate(raw_params)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid `mini_swe_agent` workflow parameters: {exc}",
                error_code="CFG-002",
            ) from exc

        if settings.step_limit is None:
            settings = settings.model_copy(
                update={"step_limit": int(config.agent_runtime.max_steps)}
            )
        if settings.wall_time_limit_seconds is None:
            settings = settings.model_copy(
                update={"wall_time_limit_seconds": int(config.sandbox.timeout)}
            )
        if config.sandbox.backend not in {"local", "docker"}:
            raise ConfigurationError(
                "mini_swe_agent supports only sandbox.backend='local' or "
                f"'docker'; got {config.sandbox.backend!r}.",
                error_code="CFG-002",
            )

        workspace = WorkspaceService(
            run_name=config.run.run_name,
            base_dir=settings.workspace_base_dir,
        )
        telemetry = MiniSWEAgentTelemetry()
        services = {
            "llm": telemetry,
            "workspace": workspace,
        }
        return MiniSWEAgentWorkflow(
            operators={},
            services=services,
            agent_config=config.agent.model_dump(),
            llm_config=config.llm,
            settings=settings,
            sandbox_config=config.sandbox,
        )


__all__ = ["MiniSWEAgentWorkflowFactory"]
