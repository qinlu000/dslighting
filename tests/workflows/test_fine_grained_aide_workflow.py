"""Unit tests for the current fine-grained AIDE DAG actor contract."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from dslighting.runtime.dag import NodeResult
from dslighting.workflows.search.aide_workflow import FineGrainedAIDEWorkflowDagActor


@pytest.fixture
def actor():
    workflow = Mock()
    workflow.agent_config = {"search": {"max_iterations": 3}}
    workflow.visualization_policy = "allow"
    workflow.state.generate_summary.return_value = ""
    workflow._select_node_to_expand.return_value = None
    workflow.generate_op = AsyncMock(return_value=("test_plan", "test_code"))
    workflow.execute_op = AsyncMock()
    workflow.review_op = AsyncMock()
    workflow._finalize_best_solution = AsyncMock(return_value={"status": "success"})

    return FineGrainedAIDEWorkflowDagActor(
        task_id="test_task",
        workflow=workflow,
        description="Test task description",
        io_instructions="Test I/O instructions",
        output_path=Path("/tmp/test_output"),
        max_retries=3,
    )


def _result(node_id: str, *, status: str = "success", outputs=None, error=None):
    return NodeResult(
        node_id=node_id,
        task_id="test_task",
        status=status,
        outputs=outputs or {},
        error=error,
    )


def test_initial_node_is_first_generation(actor):
    nodes = actor.initial_nodes()

    assert len(nodes) == 1
    assert nodes[0].node_id == "test_task:gen:0"
    assert nodes[0].depends_on == []


def test_generation_outputs_flow_to_execution(actor):
    result = _result(
        "test_task:gen:0",
        outputs={"plan": "my_plan", "code": "my_code"},
    )

    nodes, done = actor.on_node_result(result)

    assert not done
    assert len(nodes) == 1
    assert nodes[0].node_id == "test_task:exec:0"
    assert nodes[0].depends_on == [result.node_id]
    assert nodes[0].payload["kwargs"]["code"] == "my_code"


def test_empty_generation_advances_to_next_search_iteration(actor):
    result = _result(
        "test_task:gen:0",
        outputs={"plan": "plan", "code": ""},
    )

    nodes, done = actor.on_node_result(result)

    assert not done
    assert nodes[0].node_id == "test_task:gen:1"
    assert nodes[0].depends_on == [result.node_id]


def test_generation_failure_stops_after_retry_budget(actor):
    result = _result(
        "test_task:gen:0",
        status="failed",
        error="Generation failed",
    )

    first_nodes, first_done = actor.on_node_result(result)
    second_nodes, second_done = actor.on_node_result(result)
    final_nodes, final_done = actor.on_node_result(result)

    assert not first_done and first_nodes[0].node_id == "test_task:gen:1"
    assert not second_done and second_nodes[0].node_id == "test_task:gen:1"
    assert final_done and final_nodes == []
    assert actor.get_result()["status"] == "failed"


def test_execution_outputs_flow_to_review(actor):
    result = _result(
        "test_task:exec:0",
        outputs={
            "success": True,
            "stdout": "Execution successful",
            "stderr": "",
            "exc_type": None,
            "metadata": {"time": 1.0},
            "code": "executed_code",
        },
    )

    nodes, done = actor.on_node_result(result)

    assert not done
    assert nodes[0].node_id == "test_task:review:0"
    assert nodes[0].depends_on == [result.node_id]
    prompt_context = nodes[0].payload["kwargs"]["prompt_context"]
    assert prompt_context["code"] == "executed_code"
    assert prompt_context["output"] == "Execution successful"


def test_review_creates_next_generation_with_dependency(actor):
    result = _result("test_task:review:0")

    nodes, done = actor.on_node_result(result)

    assert not done
    assert nodes[0].node_id == "test_task:gen:1"
    assert nodes[0].depends_on == [result.node_id]


def test_final_review_creates_finalize_node(actor):
    result = _result("test_task:review:2")

    nodes, done = actor.on_node_result(result)

    assert not done
    assert nodes[0].node_id == "test_task:finalize"
    assert nodes[0].depends_on == [result.node_id]


def test_execution_failure_advances_to_next_search_iteration(actor):
    result = _result(
        "test_task:exec:0",
        status="failed",
        error="Execution failed",
    )

    nodes, done = actor.on_node_result(result)

    assert not done
    assert nodes[0].node_id == "test_task:gen:1"
    assert nodes[0].depends_on == [result.node_id]
