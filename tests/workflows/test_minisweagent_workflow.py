from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dslighting.api.agent import Agent
from dslighting.config import (
    DSLightingConfig,
    LLMConfig,
    SandboxConfig,
    WorkflowConfig,
)
from dslighting.core.application.agent_config_builder import AgentConfigBuilder
from dslighting.core.config import ConfigBuilder
from dslighting.error import ConfigurationError, WorkflowError
from dslighting.services.workspace import WorkspaceService
from dslighting.workflows.factory.minisweagent import MiniSWEAgentWorkflowFactory
from dslighting.workflows.factory.registry import default_workflow_registry
from dslighting.workflows.search.minisweagent.workflow import (
    MiniSWEAgentSettings,
    MiniSWEAgentTelemetry,
    MiniSWEAgentWorkflow,
    _RedactingModelProxy,
)


def _agent_config_builder(**init_kwargs) -> AgentConfigBuilder:
    return AgentConfigBuilder(
        workflow_name="mini_swe_agent",
        model="openai/gpt-4o",
        api_key="test-key",
        api_keys=None,
        api_base=None,
        provider=None,
        temperature=None,
        timeout=300,
        keep_workspace=False,
        sandbox_backend=None,
        sandbox_backend_type=None,
        sandbox_timeout=None,
        sandbox_api_key=None,
        init_kwargs=init_kwargs,
    )


@pytest.mark.parametrize(
    "alias",
    ["mini_swe_agent", "mini-swe-agent", "minisweagent", "mini"],
)
def test_agent_accepts_minisweagent_aliases(alias: str) -> None:
    agent = Agent(workflow=alias, model="openai/gpt-4o", api_key="test-key")
    assert agent.workflow_name == "mini_swe_agent"


def test_agent_config_builder_maps_minisweagent_namespace() -> None:
    config = _agent_config_builder(mini_swe_agent={"step_limit": 8, "command_timeout": 45}).build(
        task_id="demo",
        run_kwargs={"mini_swe_agent": {"step_limit": 12, "cost_limit": 1.5}},
    )

    assert config.workflow is not None
    assert config.workflow.name == "mini_swe_agent"
    assert config.workflow.params == {
        "step_limit": 12,
        "cost_limit": 1.5,
    }


def test_config_builder_maps_minisweagent_namespace() -> None:
    config = ConfigBuilder().build_config(
        workflow="mini_swe_agent",
        model="openai/gpt-4o",
        mini_swe_agent={"step_limit": 7, "command_timeout": 60},
    )

    assert config.workflow is not None
    assert config.workflow.params == {
        "step_limit": 7,
        "command_timeout": 60,
    }


def test_registry_resolves_minisweagent_factory() -> None:
    factory = default_workflow_registry.resolve("mini_swe_agent")
    assert isinstance(factory, MiniSWEAgentWorkflowFactory)


def test_factory_uses_shared_runtime_defaults(monkeypatch, tmp_path: Path) -> None:
    import dslighting.workflows.factory.minisweagent as factory_module

    class DummyWorkspace:
        def __init__(self, run_name: str, base_dir: str | None = None):
            self.run_name = run_name
            self.base_dir = base_dir

    monkeypatch.setattr(factory_module, "WorkspaceService", DummyWorkspace)

    config = DSLightingConfig(
        workflow=WorkflowConfig(
            name="mini_swe_agent",
            params={
                "workspace_base_dir": str(tmp_path),
                "command_timeout": 90,
            },
        ),
        llm=LLMConfig(model="openai/gpt-4o", api_key="test-key"),
    )
    config.agent_runtime.max_steps = 14
    config.sandbox.timeout = 600

    workflow = MiniSWEAgentWorkflowFactory().create_workflow(config)

    assert workflow.settings.step_limit == 14
    assert workflow.settings.wall_time_limit_seconds == 600
    assert workflow.settings.command_timeout == 90
    assert workflow.workspace_service.base_dir == str(tmp_path)
    assert workflow.sandbox_config.backend == "local"
    assert "sandbox" not in workflow.services
    assert workflow.services["llm"] is workflow.telemetry


def test_factory_delegates_docker_to_official_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import dslighting.workflows.factory.minisweagent as factory_module

    class DummyWorkspace:
        def __init__(self, run_name: str, base_dir: str | None = None):
            self.run_name = run_name
            self.base_dir = base_dir

    monkeypatch.setattr(factory_module, "WorkspaceService", DummyWorkspace)

    config = DSLightingConfig(
        workflow=WorkflowConfig(name="mini_swe_agent"),
        llm=LLMConfig(model="openai/gpt-4o", api_key="test-key"),
        sandbox=SandboxConfig(
            backend="docker",
            docker_image="benchmark@sha256:abc",
        ),
    )

    workflow = MiniSWEAgentWorkflowFactory().create_workflow(config)

    assert "sandbox" not in workflow.services
    assert workflow.sandbox_config is config.sandbox
    assert workflow.sandbox_config.docker_image == "benchmark@sha256:abc"


@pytest.mark.parametrize("backend", ["e2b", "ds_sandbox"])
def test_factory_rejects_non_official_environment_backends(
    backend: str,
) -> None:
    config = DSLightingConfig(
        workflow=WorkflowConfig(name="mini_swe_agent"),
        llm=LLMConfig(model="openai/gpt-4o", api_key="test-key"),
        sandbox=SandboxConfig(backend=backend),
    )

    with pytest.raises(ConfigurationError, match="supports only.*local.*docker"):
        MiniSWEAgentWorkflowFactory().create_workflow(config)


class _DummyWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "sandbox_workdir").mkdir(parents=True)
        (root / "artifacts").mkdir(parents=True)

    def get_path(self, name: str) -> Path:
        return self.root / name


def test_official_docker_config_mounts_workspace_and_inputs(
    tmp_path: Path,
) -> None:
    workflow, workspace = _workflow(tmp_path)
    source_dir = tmp_path / "benchmark-data"
    source_dir.mkdir()
    source_file = source_dir / "data.csv"
    source_file.write_text("x\n1\n", encoding="utf-8")
    (workspace.get_path("sandbox_workdir") / "data.csv").symlink_to(source_file)
    workflow.sandbox_config = SandboxConfig(
        backend="docker",
        docker_image="agenticdatabench@sha256:abc",
        docker_workspace_path="/benchmark",
        network_policy="disabled",
        timeout=600,
        memory_mb=8192,
        cpu_cores=4.0,
        pids_limit=128,
    )

    config = workflow._build_official_docker_config()

    assert config["environment_class"] == "docker"
    assert config["image"] == "agenticdatabench@sha256:abc"
    assert config["cwd"] == "/benchmark"
    assert config["interpreter"] == ["bash", "-lc"]
    assert config["timeout"] == workflow.settings.command_timeout
    assert config["env"]["HOME"] == "/benchmark/.sandbox_home"
    run_args = config["run_args"]
    assert run_args[:2] == ["--rm", "--mount"]
    assert "--network" in run_args
    assert run_args[run_args.index("--network") + 1] == "none"
    assert "--memory" in run_args
    assert run_args[run_args.index("--memory") + 1] == "8192m"
    mount_specs = [
        run_args[index + 1] for index, value in enumerate(run_args) if value == "--mount"
    ]
    assert any(
        f"src={workspace.get_path('sandbox_workdir').resolve()}" in value
        and "dst=/benchmark" in value
        and "readonly" not in value
        for value in mount_specs
    )
    assert any(
        f"src={source_file.resolve()}" in value
        and f"dst={source_file}" in value
        and value.endswith(",readonly")
        for value in mount_specs
    )


class _FakeAgent:
    def __init__(self, output: Path | None) -> None:
        self.output = output
        self.cost = 0.25
        self.messages: list[dict] = []
        self.prompt = ""
        self.thread_id: int | None = None

    def run(self, prompt: str) -> dict:
        self.prompt = prompt
        self.thread_id = threading.get_ident()
        if self.output is not None:
            self.output.write_text("prediction\n1\n", encoding="utf-8")
        self.messages = [
            {
                "role": "assistant",
                "content": "working",
                "extra": {
                    "cost": 0.25,
                    "timestamp": 1.0,
                    "response": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                        },
                        "api_key": "must-not-leak",
                    },
                },
            },
            {
                "role": "exit",
                "content": "done",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "done",
                },
            },
        ]
        return {"exit_status": "Submitted", "submission": "done"}


def _workflow(tmp_path: Path) -> tuple[MiniSWEAgentWorkflow, _DummyWorkspace]:
    workspace = _DummyWorkspace(tmp_path / "workspace")
    telemetry = MiniSWEAgentTelemetry()
    workflow = MiniSWEAgentWorkflow(
        operators={},
        services={
            "llm": telemetry,
            "workspace": workspace,
        },
        agent_config={},
        llm_config=LLMConfig(model="openai/gpt-4o", api_key="test-key"),
        settings=MiniSWEAgentSettings(
            step_limit=5,
            wall_time_limit_seconds=60,
        ),
    )
    return workflow, workspace


def test_official_local_config_uses_task_workspace(tmp_path: Path) -> None:
    workflow, workspace = _workflow(tmp_path)

    config = workflow._build_official_local_config()

    assert config == {
        "environment_class": "local",
        "cwd": str(workspace.get_path("sandbox_workdir").resolve()),
        "timeout": workflow.settings.command_timeout,
        "env": {
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
    }


@pytest.mark.asyncio
async def test_workflow_runs_agent_off_loop_and_captures_telemetry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workflow, workspace = _workflow(tmp_path)
    fake_agent = _FakeAgent(workspace.get_path("sandbox_workdir") / "submission.csv")
    monkeypatch.setattr(workflow, "_build_agent", lambda **kwargs: fake_agent)
    event_loop_thread = threading.get_ident()

    await workflow.solve(
        description="Predict the target.",
        io_instructions="Write a CSV submission.",
        data_dir=tmp_path / "data",
        output_path=tmp_path / "out" / "submission.csv",
    )

    assert fake_agent.thread_id != event_loop_thread
    assert "Predict the target." in fake_agent.prompt
    assert "exact relative path `submission.csv`" in fake_agent.prompt
    assert workflow.last_run_result["exit_status"] == "Submitted"
    assert workflow.telemetry.get_total_cost() == pytest.approx(0.25)
    assert workflow.telemetry.get_usage_summary()["total_tokens"] == 15
    call = workflow.telemetry.get_call_history()[0]
    assert call["response"]["api_key"] == "***REDACTED***"
    assert call["response"]["usage"]["prompt_tokens"] == 10
    assert call["response"]["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_workflow_fails_when_required_output_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workflow, _ = _workflow(tmp_path)
    fake_agent = _FakeAgent(None)
    monkeypatch.setattr(workflow, "_build_agent", lambda **kwargs: fake_agent)

    with pytest.raises(WorkflowError, match="without creating the required output"):
        await workflow.solve(
            description="Predict the target.",
            io_instructions="Write a CSV submission.",
            data_dir=tmp_path / "data",
            output_path=tmp_path / "out" / "submission.csv",
        )


def test_model_proxy_redacts_serialized_credentials() -> None:
    underlying = SimpleNamespace(
        config=SimpleNamespace(),
        serialize=lambda: {
            "model": {
                "model_kwargs": {
                    "api_key": "secret",
                    "nested": {"access_token": "also-secret"},
                }
            }
        },
    )
    serialized = _RedactingModelProxy(underlying).serialize()

    assert serialized["model"]["model_kwargs"]["api_key"] == "***REDACTED***"
    assert serialized["model"]["model_kwargs"]["nested"]["access_token"] == "***REDACTED***"


@pytest.mark.asyncio
async def test_official_v2_agent_runs_end_to_end_offline(tmp_path: Path) -> None:
    pytest.importorskip("minisweagent")

    workspace = WorkspaceService(run_name="mini-e2e", base_dir=str(tmp_path))
    telemetry = MiniSWEAgentTelemetry()
    outputs = [
        {
            "role": "assistant",
            "content": "Create the required artifact.",
            "extra": {
                "actions": [{"command": ("printf 'prediction\\n1\\n' > submission.csv")}],
                "cost": 0.1,
                "timestamp": 1.0,
            },
        },
        {
            "role": "assistant",
            "content": "Submit the completed task.",
            "extra": {
                "actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
                "cost": 0.1,
                "timestamp": 2.0,
            },
        },
    ]
    workflow = MiniSWEAgentWorkflow(
        operators={},
        services={
            "llm": telemetry,
            "workspace": workspace,
        },
        agent_config={},
        llm_config=LLMConfig(model="deterministic"),
        settings=MiniSWEAgentSettings(
            step_limit=3,
            wall_time_limit_seconds=30,
            command_timeout=10,
            model_overrides={
                "model_class": "deterministic",
                "outputs": outputs,
            },
        ),
        sandbox_config=SandboxConfig(backend="local", timeout=10),
    )

    await workflow.solve(
        description="Create a deterministic submission.",
        io_instructions="Write prediction rows.",
        data_dir=tmp_path / "data",
        output_path=tmp_path / "out" / "submission.csv",
    )

    generated = workspace.get_path("sandbox_workdir") / "submission.csv"
    assert generated.read_text(encoding="utf-8") == "prediction\n1\n"
    assert workflow.last_run_result["exit_status"] == "Submitted"
    assert telemetry.get_total_cost() == pytest.approx(0.2)
    assert (workspace.get_path("artifacts") / "minisweagent_trajectory.json").exists()
