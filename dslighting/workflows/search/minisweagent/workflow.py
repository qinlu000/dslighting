"""mini-swe-agent v2 adapter for DSLighting.

The integration keeps DSLighting in charge of task preparation, workspace
management, benchmark output collection, and telemetry.  mini-swe-agent
provides the agent loop, model implementation, and official Local/Docker
command environments.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from pathlib import Path
from typing import Any, Mapping, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dslighting.config import LLMConfig, SandboxConfig
from dslighting.error import ConfigurationError, WorkflowError
from dslighting.services.sandbox_backends.backends.docker import (
    resolve_docker_input_mounts,
)
from dslighting.workflows.base import BaseWorkflow

logger = logging.getLogger(__name__)

_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|cookie|credentials?|password|passwd|"
    r"private[_-]?key|access[_-]?key|secret|"
    r"(?:access|auth|bearer|refresh|session)[_-]?token)(?:$|[_-])|^token$",
    re.IGNORECASE,
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _redact(value: Any) -> Any:
    """Return a JSON-friendly copy with credentials removed."""
    if isinstance(value, Mapping):
        return {
            str(key): "***REDACTED***" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class MiniSWEAgentSettings(BaseModel):
    """Validated ``workflow.params`` accepted by the integration."""

    model_config = ConfigDict(extra="forbid")

    workspace_base_dir: str | None = None
    config_path: str | None = None
    step_limit: int | None = Field(default=None, ge=0)
    cost_limit: float = Field(default=0.0, ge=0.0)
    wall_time_limit_seconds: int | None = Field(default=None, ge=0)
    command_timeout: int = Field(default=30, gt=0)
    agent_overrides: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reserved_overrides(self) -> "MiniSWEAgentSettings":
        reserved_agent = {
            "output_path",
            "step_limit",
            "cost_limit",
            "wall_time_limit_seconds",
        }
        invalid_agent = sorted(reserved_agent.intersection(self.agent_overrides))
        if invalid_agent:
            raise ValueError(
                "Set managed agent options directly under `mini_swe_agent`; "
                f"do not put them in `agent_overrides`: {invalid_agent}"
            )

        reserved_model = {"model_name"}
        invalid_model = sorted(reserved_model.intersection(self.model_overrides))
        if invalid_model:
            raise ValueError(
                "The model name is managed by `llm.model`; "
                f"do not put it in `model_overrides`: {invalid_model}"
            )
        return self


class _RedactingModelProxy:
    """Delegate mini-swe-agent's model protocol while sanitizing trajectories."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.config = model.config

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return cast(dict[str, Any], self._model.query(messages, **kwargs))

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return cast(dict[str, Any], self._model.format_message(**kwargs))

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._model.format_observation_messages(message, outputs, template_vars),
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return cast(dict[str, Any], self._model.get_template_vars(**kwargs))

    def serialize(self) -> dict[str, Any]:
        return cast(dict[str, Any], _redact(self._model.serialize()))


class MiniSWEAgentTelemetry:
    """Small adapter exposing mini-swe-agent usage through DSLighting's telemetry API."""

    def __init__(self) -> None:
        self._total_cost = 0.0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_history: list[dict[str, Any]] = []
        self.exit_status = ""

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def capture(self, agent: Any) -> None:
        """Capture a completed (or failed) agent without retaining credentials."""
        self._total_cost = float(getattr(agent, "cost", 0.0) or 0.0)
        self._prompt_tokens = 0
        self._completion_tokens = 0
        history: list[dict[str, Any]] = []

        for message in getattr(agent, "messages", []) or []:
            extra = message.get("extra", {}) if isinstance(message, dict) else {}
            response = extra.get("response")
            if isinstance(response, Mapping):
                usage = response.get("usage", {})
                if isinstance(usage, Mapping):
                    self._prompt_tokens += self._coerce_int(
                        usage.get("prompt_tokens") or usage.get("input_tokens")
                    )
                    self._completion_tokens += self._coerce_int(
                        usage.get("completion_tokens") or usage.get("output_tokens")
                    )
                history.append(
                    _redact(
                        {
                            "timestamp": extra.get("timestamp"),
                            "cost": extra.get("cost", 0.0),
                            "response": response,
                        }
                    )
                )

        self._call_history = history
        messages = getattr(agent, "messages", []) or []
        if messages and isinstance(messages[-1], dict):
            self.exit_status = str(messages[-1].get("extra", {}).get("exit_status", ""))

    def get_total_cost(self) -> float:
        return self._total_cost

    def get_call_history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._call_history)

    def get_usage_summary(self) -> dict[str, Any]:
        total_tokens = self._prompt_tokens + self._completion_tokens
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_cost": None,
            "completion_tokens_cost": None,
            "total_cost": round(self._total_cost, 12),
            "call_count": len(self._call_history),
            "cost_per_token": (self._total_cost / total_tokens if total_tokens else None),
            "exit_status": self.exit_status,
            "agent": "mini-swe-agent",
        }


class MiniSWEAgentWorkflow(BaseWorkflow):
    """Run mini-swe-agent v2 against a DSLighting task workspace."""

    def __init__(
        self,
        operators: dict[str, Any],
        services: dict[str, Any],
        agent_config: dict[str, Any],
        *,
        llm_config: LLMConfig,
        settings: MiniSWEAgentSettings,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        super().__init__(operators, services, agent_config)
        self.llm_config = llm_config
        self.settings = settings
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.workspace_service = services["workspace"]
        self.telemetry: MiniSWEAgentTelemetry = services["llm"]
        self.last_run_result: dict[str, Any] = {}

    @staticmethod
    def _load_bindings() -> tuple[Any, Any, Any, Any]:
        try:
            import minisweagent
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.config import get_config_from_spec
            from minisweagent.environments import get_environment
            from minisweagent.models import get_model
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("minisweagent"):
                raise ConfigurationError(
                    "The mini-swe-agent workflow requires the optional dependency. "
                    "Install it with `pip install 'dslighting[mini-swe-agent]'`.",
                    error_code="CFG-002",
                ) from exc
            raise

        version = str(getattr(minisweagent, "__version__", ""))
        if not version.startswith("2."):
            raise ConfigurationError(
                "DSLighting's mini-swe-agent integration requires mini-swe-agent v2 "
                f"(installed version: {version or 'unknown'}).",
                error_code="CFG-002",
            )
        return DefaultAgent, get_config_from_spec, get_environment, get_model

    @staticmethod
    def _docker_mount_spec(source: Path, target: Path, *, read_only: bool) -> str:
        source_text = str(source)
        target_text = str(target)
        if any("\x00" in value or "," in value for value in (source_text, target_text)):
            raise ConfigurationError(
                "mini-swe-agent Docker bind-mount paths cannot contain NUL or commas: "
                f"{source_text!r} -> {target_text!r}",
                error_code="CFG-002",
            )
        options = ",readonly" if read_only else ""
        return f"type=bind,src={source_text},dst={target_text}{options}"

    def _build_official_docker_config(self) -> dict[str, Any]:
        sandbox = self.sandbox_config
        if sandbox.backend != "docker":
            raise ConfigurationError(
                "Official mini-swe-agent Docker execution requires sandbox.backend='docker'.",
                error_code="CFG-002",
            )

        image = str(sandbox.docker_image or "").strip()
        if not image:
            raise ConfigurationError(
                "sandbox.backend='docker' requires sandbox.docker_image",
                error_code="CFG-002",
            )

        workspace = self.workspace_service.get_path("sandbox_workdir").resolve()
        container_workspace = Path(str(sandbox.docker_workspace_path))
        if not container_workspace.is_absolute() or container_workspace == Path("/"):
            raise ConfigurationError(
                "sandbox.docker_workspace_path must be an absolute non-root path",
                error_code="CFG-002",
            )

        (workspace / ".sandbox_home" / ".config" / "matplotlib").mkdir(
            parents=True,
            exist_ok=True,
        )
        (workspace / ".sandbox_tmp").mkdir(parents=True, exist_ok=True)

        run_args = [
            "--rm",
            "--mount",
            self._docker_mount_spec(
                workspace,
                container_workspace,
                read_only=False,
            ),
            "--memory",
            f"{int(sandbox.memory_mb)}m",
            "--cpus",
            str(float(sandbox.cpu_cores)),
            "--pids-limit",
            str(int(sandbox.pids_limit)),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--label",
            "dslighting.sandbox=true",
            "--label",
            f"dslighting.image={image}",
        ]
        if sandbox.network_policy == "disabled":
            run_args.extend(["--network", "none"])
        if sandbox.docker_user:
            run_args.extend(["--user", str(sandbox.docker_user)])

        for source, visible_target in resolve_docker_input_mounts(workspace):
            run_args.extend(
                [
                    "--mount",
                    self._docker_mount_spec(
                        source,
                        visible_target,
                        read_only=True,
                    ),
                ]
            )

        workspace_text = str(container_workspace)
        container_lifetime = max(
            int(sandbox.timeout),
            int(self.settings.wall_time_limit_seconds or 0),
            60,
        )
        return {
            "environment_class": "docker",
            "image": image,
            "cwd": workspace_text,
            "timeout": int(self.settings.command_timeout),
            "container_timeout": f"{container_lifetime + 60}s",
            "interpreter": ["bash", "-lc"],
            "run_args": run_args,
            "env": {
                "HOME": f"{workspace_text}/.sandbox_home",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": (f"{workspace_text}/.sandbox_home/.config/matplotlib"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "TEMP": f"{workspace_text}/.sandbox_tmp",
                "TMP": f"{workspace_text}/.sandbox_tmp",
                "TMPDIR": f"{workspace_text}/.sandbox_tmp",
            },
        }

    def _build_official_local_config(self) -> dict[str, Any]:
        if self.sandbox_config.backend != "local":
            raise ConfigurationError(
                "Official mini-swe-agent local execution requires sandbox.backend='local'.",
                error_code="CFG-002",
            )
        logger.warning(
            "mini-swe-agent is using official LocalEnvironment; commands run "
            "directly on the host. Use sandbox.backend='docker' for benchmark runs."
        )
        return {
            "environment_class": "local",
            "cwd": str(self.workspace_service.get_path("sandbox_workdir").resolve()),
            "timeout": int(self.settings.command_timeout),
            "env": {
                "MPLBACKEND": "Agg",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        }

    def _build_environment(
        self,
        *,
        get_environment: Any,
    ) -> Any:
        if self.sandbox_config.backend == "docker":
            return get_environment(config=self._build_official_docker_config())
        if self.sandbox_config.backend == "local":
            return get_environment(config=self._build_official_local_config())
        raise ConfigurationError(
            "mini-swe-agent supports only sandbox.backend='local' or 'docker'; "
            f"got {self.sandbox_config.backend!r}.",
            error_code="CFG-002",
        )

    def _build_model_config(self, base_config: Mapping[str, Any]) -> dict[str, Any]:
        model_config = _deep_merge(base_config, self.settings.model_overrides)
        model_config["model_name"] = self.llm_config.model

        model_kwargs = dict(model_config.get("model_kwargs") or {})
        api_keys = self.llm_config.get_api_keys()
        if api_keys:
            model_kwargs["api_key"] = api_keys[0]
            if len(api_keys) > 1:
                logger.warning(
                    "mini-swe-agent uses the first configured API key; DSLighting's "
                    "key-pool rotation is not available for this workflow."
                )
        if self.llm_config.api_base:
            model_kwargs["api_base"] = self.llm_config.api_base
        if self.llm_config.provider:
            model_kwargs["custom_llm_provider"] = self.llm_config.provider
        model_kwargs["timeout"] = self.llm_config.request_timeout_seconds
        model_kwargs["num_retries"] = self.llm_config.sdk_max_retries
        if self.llm_config.thinking is not None:
            extra_body = dict(model_kwargs.get("extra_body") or {})
            extra_body["thinking"] = {"type": "enabled" if self.llm_config.thinking else "disabled"}
            model_kwargs["extra_body"] = extra_body
        model_config["model_kwargs"] = model_kwargs
        return model_config

    def _build_agent(
        self,
        *,
        trajectory_path: Path,
    ) -> Any:
        (
            default_agent_class,
            get_config_from_spec,
            get_environment,
            get_model,
        ) = self._load_bindings()
        config_spec = self.settings.config_path or "mini.yaml"
        try:
            loaded_config = get_config_from_spec(config_spec)
        except Exception as exc:
            raise ConfigurationError(
                f"Unable to load mini-swe-agent config {config_spec!r}: {exc}",
                error_code="CFG-002",
            ) from exc
        if not isinstance(loaded_config, Mapping):
            raise ConfigurationError(
                f"mini-swe-agent config {config_spec!r} must contain a YAML mapping.",
                error_code="CFG-002",
            )

        agent_config = _deep_merge(
            loaded_config.get("agent", {}),
            self.settings.agent_overrides,
        )
        agent_config.update(
            {
                "step_limit": int(self.settings.step_limit or 0),
                "cost_limit": float(self.settings.cost_limit),
                "wall_time_limit_seconds": int(self.settings.wall_time_limit_seconds or 0),
                "output_path": trajectory_path,
            }
        )
        model_config = self._build_model_config(loaded_config.get("model", {}))
        model = _RedactingModelProxy(get_model(config=model_config))
        environment = self._build_environment(
            get_environment=get_environment,
        )
        try:
            return default_agent_class(model, environment, **agent_config)
        except Exception:
            cleanup = getattr(environment, "cleanup", None)
            if callable(cleanup):
                cleanup()
            raise

    @staticmethod
    async def _cleanup_agent_environment(agent: Any) -> None:
        environment = getattr(agent, "env", None)
        cleanup = getattr(environment, "cleanup", None)
        if callable(cleanup):
            await asyncio.to_thread(cleanup)

    @staticmethod
    def _build_task_prompt(
        *,
        description: str,
        io_instructions: str,
        output_name: str,
    ) -> str:
        return (
            "Solve this DSLighting data-science task.\n\n"
            "## Task description\n"
            f"{description.strip()}\n\n"
            "## I/O requirements\n"
            f"{io_instructions.strip()}\n\n"
            "## Execution contract\n"
            "- The current working directory is the task workspace.\n"
            "- Input data files are already available in the current working directory.\n"
            f"- Create the final artifact at the exact relative path `{output_name}`.\n"
            "- Do not submit completion until that artifact exists and satisfies the I/O requirements.\n"
        )

    async def solve(
        self,
        description: str,
        io_instructions: str,
        data_dir: Path,
        output_path: Path,
    ) -> None:
        _ = data_dir
        trajectory_path = (
            self.workspace_service.get_path("artifacts") / "minisweagent_trajectory.json"
        )
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = self._build_task_prompt(
            description=description,
            io_instructions=io_instructions,
            output_name=output_path.name,
        )

        logger.info(
            "Starting mini-swe-agent | model=%s step_limit=%s command_timeout=%ss environment=%s",
            self.llm_config.model,
            self.settings.step_limit,
            self.settings.command_timeout,
            f"official_{self.sandbox_config.backend}",
        )
        agent = None
        try:
            agent = await asyncio.to_thread(
                self._build_agent,
                trajectory_path=trajectory_path,
            )
            result = await asyncio.to_thread(agent.run, prompt)
            self.last_run_result = dict(result or {})
        finally:
            if agent is not None:
                self.telemetry.capture(agent)
                await self._cleanup_agent_environment(agent)

        sandbox_output = self.workspace_service.get_path("sandbox_workdir") / output_path.name
        if not sandbox_output.exists():
            exit_status = self.last_run_result.get("exit_status") or self.telemetry.exit_status
            raise WorkflowError(
                "mini-swe-agent finished without creating the required output "
                f"`{output_path.name}` (exit_status={exit_status or 'unknown'}).",
                error_code="WRK-003",
                details={
                    "output_name": output_path.name,
                    "exit_status": exit_status,
                    "trajectory_path": str(trajectory_path),
                },
                suggestion=(
                    "Increase `mini_swe_agent.step_limit` or inspect the saved trajectory."
                ),
            )

        logger.info(
            "mini-swe-agent completed | exit_status=%s cost=$%.6f output=%s",
            self.last_run_result.get("exit_status") or self.telemetry.exit_status,
            self.telemetry.get_total_cost(),
            sandbox_output,
        )


__all__ = [
    "MiniSWEAgentSettings",
    "MiniSWEAgentTelemetry",
    "MiniSWEAgentWorkflow",
]
