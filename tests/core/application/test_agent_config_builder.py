import pytest

from dslighting.config import LLMConfig
from dslighting.core.application.agent_config_builder import AgentConfigBuilder
from dslighting.core.visualization_policy import VisualizationPolicy
from dslighting.error import ConfigurationError


def _make_builder(workflow_name: str, init_kwargs: dict):
    return AgentConfigBuilder(
        workflow_name=workflow_name,
        llm_config=LLMConfig(model="gpt-4o"),
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


def test_removed_data_card_context_is_rejected() -> None:
    builder = _make_builder("aide", {})

    with pytest.raises(ConfigurationError, match="Data Card"):
        builder.build(task_id="task1", run_kwargs={"task_context": {"policy": "l1"}})
