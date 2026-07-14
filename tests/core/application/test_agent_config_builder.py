import pytest

from dslighting.core.application.agent_config_builder import AgentConfigBuilder
from dslighting.core.visualization_policy import VisualizationPolicy
from dslighting.error import ConfigurationError


def _make_builder(workflow_name: str, init_kwargs: dict):
    return AgentConfigBuilder(
        workflow_name=workflow_name,
        model="gpt-4o",
        api_key=None,
        api_keys=None,
        api_base=None,
        provider=None,
        temperature=None,
        timeout=300,
        keep_workspace=True,
        sandbox_backend=None,
        sandbox_backend_type=None,
        sandbox_timeout=None,
        sandbox_api_key=None,
        init_kwargs=init_kwargs,
    )


def test_visualization_policy_goes_to_shared_agent_config() -> None:
    builder = _make_builder("aide", {"max_iterations": 2})
    config = builder.build(task_id="task1", run_kwargs={"visualization_policy": "allow"})

    assert config.agent.search.max_iterations == 2
    assert config.agent.visualization.policy == VisualizationPolicy.ALLOW


def test_legacy_enforce_no_plotting_maps_to_no_display_policy() -> None:
    builder = _make_builder("autokaggle", {})
    config = builder.build(task_id="task1", run_kwargs={"enforce_no_plotting": False})

    assert config.agent.visualization.policy == VisualizationPolicy.ALLOW


def test_runtime_kwargs_override_init_kwargs_and_unconsumed_go_to_parameters() -> None:
    builder = _make_builder("aide", {"max_iterations": 2, "custom_a": 1})
    config = builder.build(task_id="task1", run_kwargs={"max_iterations": 5, "custom_b": 2})

    assert config.agent.search.max_iterations == 5
    assert config.run.parameters["custom_a"] == 1
    assert config.run.parameters["custom_b"] == 2


def test_task_context_runtime_kwargs_override_init_policy() -> None:
    builder = _make_builder(
        "aide",
        {"task_context": {"policy": "l1", "l1_artifact_dir": "/annotations"}},
    )

    config = builder.build(
        task_id="task1",
        run_kwargs={
            "task_context": {
                "policy": "l2",
                "l2_guidance_path": "/guidance.md",
            }
        },
    )

    assert config.task_context.policy == "l2"
    assert config.task_context.l1_artifact_dir is None
    assert config.task_context.l2_guidance_path == "/guidance.md"
    assert "task_context" not in config.run.parameters


@pytest.mark.parametrize("value", ["not-a-dict", {"policy": "l0"}])
def test_agent_config_builder_rejects_invalid_task_context(value: object) -> None:
    builder = _make_builder("aide", {})

    with pytest.raises((ConfigurationError, ValueError)):
        builder.build(task_id="task1", run_kwargs={"task_context": value})
