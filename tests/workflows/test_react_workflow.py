from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from dslighting.config import OutputContractConfig
from dslighting.ops.presets.react import ReActOperator
from dslighting.services.sandbox_backends.backends.base import SandboxBackendConfig
from dslighting.services.sandbox_backends.backends.docker import DockerSandboxBackend
from dslighting.utils.typing import ExecutionResult
from dslighting.workflows.search.react.workflow import ReActWorkflow


class _DummyWorkspaceService:
    def __init__(self, root: Path):
        self.root = root
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.root / "sandbox_workdir").mkdir(parents=True, exist_ok=True)
        self.linked_data_dir: Path | None = None

    def get_path(self, key: str) -> Path:
        return self.root / key

    def link_data_to_workspace(self, data_dir: Path) -> None:
        self.linked_data_dir = data_dir
        sandbox_workdir = self.get_path("sandbox_workdir")
        for path in data_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, sandbox_workdir / path.name)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _DummyLLMService:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []
        self.config = SimpleNamespace(max_retries=2)

    async def call_messages(self, messages, max_retries=None):
        self.calls.append(messages)
        assert self._responses, "No more stubbed LLM responses available."
        return _DummyResponse(self._responses.pop(0))


def _offline_docker_sandbox_service() -> SimpleNamespace:
    backend = DockerSandboxBackend(
        image="agenticdatabench:test",
        config=SandboxBackendConfig(
            environment_policy="allowlist",
            network_policy="disabled",
        )
    )
    return SimpleNamespace(backend=backend)


class _FakeExecuteOperator:
    def __init__(
        self,
        workspace: _DummyWorkspaceService,
        *,
        stdout: str = "ok",
        create_output_name: str | None = None,
        create_directory_output_name: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.stdout = stdout
        self.create_output_name = create_output_name
        self.create_directory_output_name = create_directory_output_name
        self.calls: list[dict[str, object]] = []

    async def __call__(self, code: str, mode: str = "script", executor_context=None):
        self.calls.append({"code": code, "mode": mode})
        _ = executor_context

        if self.create_output_name:
            target = self.workspace.get_path("sandbox_workdir") / self.create_output_name
            target.write_text("prediction\n1\n", encoding="utf-8")

        if self.create_directory_output_name:
            submission_dir = (
                self.workspace.get_path("sandbox_workdir") / self.create_directory_output_name
            )
            submission_dir.mkdir(parents=True, exist_ok=True)
            (submission_dir / "before_covariance.csv").write_text(
                "a,b\n1,2\n",
                encoding="utf-8",
            )
            (submission_dir / "after_covariance.csv").write_text(
                "a,b\n3,4\n",
                encoding="utf-8",
            )

        return ExecutionResult(success=True, stdout=self.stdout, stderr="")


def _bind_offline_docker_sandbox(
    execute_operator: _FakeExecuteOperator,
) -> SimpleNamespace:
    sandbox = _offline_docker_sandbox_service()
    execute_operator.sandbox = sandbox
    return sandbox


@pytest.mark.asyncio
async def test_react_workflow_executes_via_shared_execute_operator_and_saves_messages(
    tmp_path,
) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "<Think>Inspect the data.</Think>\n<Action>```python\nprint('run')\n```</Action>",
            "<Think>Done.</Think>\n<Answer>final answer</Answer>",
        ]
    )

    react_operator = ReActOperator(max_steps=3)
    execute_operator = _FakeExecuteOperator(
        workspace,
        stdout="run",
        create_output_name="submission.csv",
    )
    workflow = ReActWorkflow(
        operators={"react": react_operator, "execute": execute_operator},
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    output_path = tmp_path / "out" / "submission.csv"

    await workflow.solve(
        description='Save the results to "submission.csv".',
        io_instructions="Write a submission file.",
        data_dir=data_dir,
        output_path=output_path,
    )

    assert workspace.linked_data_dir == data_dir
    sandbox_output = workspace.get_path("sandbox_workdir") / "submission.csv"
    assert sandbox_output.read_text(encoding="utf-8") == "prediction\n1\n"
    assert not output_path.exists()
    messages_path = workspace.get_path("artifacts") / "messages.json"
    assert messages_path.exists()
    saved_messages = json.loads(messages_path.read_text(encoding="utf-8"))
    assert saved_messages[0]["role"] == "system"
    assert saved_messages[2]["role"] == "assistant"
    assert execute_operator.calls == [{"code": "print('run')", "mode": "script"}]
    assert len(llm.calls) == 2
    assert llm.calls[0][0]["role"] == "system"
    assert "Role:" in llm.calls[0][0]["content"]
    assert "Task Goal and Data Overview:" not in llm.calls[0][0]["content"]
    assert "Task Description:" in llm.calls[0][1]["content"]
    assert "I/O Requirements:" in llm.calls[0][1]["content"]
    assert "exact filename `submission.csv`" not in llm.calls[0][0]["content"]
    assert "<Explore>" not in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_react_workflow_runs_perception_with_independent_context_and_returns_report(
    tmp_path,
) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            (
                "<Think>I need local evidence.</Think>"
                "<Explore>Inspect row count and missingness.</Explore>"
            ),
            (
                "<Action>```python\nprint('rows=3, missing=0')\n```</Action>"
            ),
            "<Report>rows=3; missing=0</Report>",
            "<Think>Use the evidence.</Think><Answer>final answer</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(
        workspace,
        stdout="rows=3, missing=0",
    )
    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=2),
            "execute": execute_operator,
        },
        services={
            "llm": llm,
            "sandbox": _bind_offline_docker_sandbox(execute_operator),
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )
    assert workflow.execute_op is execute_operator
    assert execute_operator.sandbox is workflow.sandbox_service

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "table.csv").write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

    await workflow.solve(
        description="Assess the claim using the local data.",
        io_instructions="Return the conclusion.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    assert execute_operator.calls == [
        {"code": "print('rows=3, missing=0')", "mode": "script"}
    ]
    assert len(llm.calls) == 4

    perception_first_call = llm.calls[1]
    assert "Perception Agent" in perception_first_call[0]["content"]
    assert perception_first_call[1]["content"] == (
        "Exploration Request:\nInspect row count and missingness."
    )
    assert "I/O Requirements:" not in perception_first_call[1]["content"]

    perception_second_call_payload = "\n".join(
        message["content"] for message in llm.calls[2]
    )
    assert "<Observation>" in perception_second_call_payload
    assert "rows=3, missing=0" in perception_second_call_payload

    solver_second_call_payload = "\n".join(message["content"] for message in llm.calls[3])
    assert "<PerceptionResult>\nrows=3; missing=0\n</PerceptionResult>" in (
        solver_second_call_payload
    )
    assert "You are a Perception Agent" not in solver_second_call_payload
    assert "print('rows=3, missing=0')" not in solver_second_call_payload

    saved_messages = json.loads(
        (workspace.get_path("artifacts") / "messages.json").read_text(encoding="utf-8")
    )
    saved_payload = "\n".join(message["content"] for message in saved_messages)
    assert "<PerceptionResult>\nrows=3; missing=0\n</PerceptionResult>" in saved_payload
    assert "print('rows=3, missing=0')" not in saved_payload


@pytest.mark.asyncio
async def test_react_workflow_rejects_nested_perception_request_without_recursing(
    tmp_path,
) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "<Think>Need evidence.</Think><Explore>Inspect rows.</Explore>",
            (
                "<Explore>Ask another Perception Agent to inspect rows.</Explore>"
            ),
            "<Report>rows=3</Report>",
            "<Think>Done.</Think><Answer>final answer</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(workspace)
    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=2),
            "execute": execute_operator,
        },
        services={
            "llm": llm,
            "sandbox": _bind_offline_docker_sandbox(execute_operator),
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    await workflow.solve(
        description="Assess the local data.",
        io_instructions="Return the conclusion.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    assert execute_operator.calls == []
    assert len(llm.calls) == 4
    assert "Perception protocol error:" in llm.calls[2][-1]["content"]
    assert "<Action>```python" in llm.calls[2][-1]["content"]
    assert "<Report>concise plain-text perception report</Report>" in (
        llm.calls[2][-1]["content"]
    )
    assert "Ask another Perception Agent" not in llm.calls[2][1]["content"]


@pytest.mark.asyncio
async def test_react_workflow_returns_bounded_failure_when_perception_exhausts_steps(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.workflows.search.react.workflow.PERCEPTION_MAX_STEPS",
        1,
    )
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "<Think>Need evidence.</Think><Explore>Inspect rows.</Explore>",
            "<Action>```python\nprint('partial')\n```</Action>",
            "<Think>Continue without it.</Think><Answer>final answer</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(workspace, stdout="partial")
    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=2),
            "execute": execute_operator,
        },
        services={
            "llm": llm,
            "sandbox": _bind_offline_docker_sandbox(execute_operator),
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    await workflow.solve(
        description="Assess the local data.",
        io_instructions="Return the conclusion.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    solver_second_call_payload = "\n".join(message["content"] for message in llm.calls[2])
    assert "Perception Agent reached its step limit" in solver_second_call_payload
    assert "partial" not in solver_second_call_payload
    assert execute_operator.calls == [
        {"code": "print('partial')", "mode": "script"}
    ]


@pytest.mark.asyncio
async def test_react_workflow_refuses_perception_when_executor_uses_different_sandbox(
    tmp_path,
) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "<Think>Need evidence.</Think><Explore>Inspect rows.</Explore>",
            "<Think>Continue safely.</Think><Answer>final answer</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(workspace)
    execute_operator.sandbox = _offline_docker_sandbox_service()
    declared_sandbox = _offline_docker_sandbox_service()
    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=2),
            "execute": execute_operator,
        },
        services={
            "llm": llm,
            "sandbox": declared_sandbox,
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    await workflow.solve(
        description="Assess the local data.",
        io_instructions="Return the conclusion.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    solver_second_call_payload = "\n".join(message["content"] for message in llm.calls[1])
    assert "<Explore> is unavailable in this workflow" in solver_second_call_payload
    assert "<Explore>...</Explore>" not in llm.calls[0][0]["content"]
    assert execute_operator.calls == []
    assert len(llm.calls) == 2


def test_react_workflow_requires_explicit_perception_opt_in(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    execute_operator = _FakeExecuteOperator(workspace)
    sandbox = _bind_offline_docker_sandbox(execute_operator)

    default_workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=1),
            "execute": execute_operator,
        },
        services={
            "llm": _DummyLLMService([]),
            "sandbox": sandbox,
            "workspace": workspace,
        },
        agent_config={},
    )
    enabled_workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=1),
            "execute": execute_operator,
        },
        services={
            "llm": _DummyLLMService([]),
            "sandbox": sandbox,
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )

    assert default_workflow._has_offline_docker_perception() is False
    assert enabled_workflow._has_offline_docker_perception() is True


@pytest.mark.asyncio
async def test_perception_loop_rechecks_offline_docker_capability(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService([])
    execute_operator = _FakeExecuteOperator(workspace)
    execute_operator.sandbox = _offline_docker_sandbox_service()
    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=1),
            "execute": execute_operator,
        },
        services={
            "llm": llm,
            "sandbox": _offline_docker_sandbox_service(),
            "workspace": workspace,
            "perception_enabled": True,
        },
        agent_config={},
    )

    report = await workflow._run_perception_loop(
        request="Inspect rows.",
        system_prompt="Perception",
    )

    assert "was not started" in report
    assert llm.calls == []
    assert execute_operator.calls == []


@pytest.mark.asyncio
async def test_react_workflow_allows_plain_answer_without_artifact_gate(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(["<Think>x</Think><Answer>42</Answer>"])

    workflow = ReActWorkflow(
        operators={"react": ReActOperator(max_steps=1), "execute": _FakeExecuteOperator(workspace)},
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "out" / "answer.txt"

    await workflow.solve(
        description="Return the final answer.",
        io_instructions="Write the answer to answer.txt.",
        data_dir=data_dir,
        output_path=output_path,
    )

    assert not output_path.exists()
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_react_workflow_rejects_final_answer_until_output_exists(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "<Think>Done.</Think><Answer>final</Answer>",
            "<Think>Create output.</Think><Action>```python\nprint('write')\n```</Action>",
            "<Think>Now done.</Think><Answer>final</Answer>",
        ]
    )

    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=3),
            "execute": _FakeExecuteOperator(
                workspace,
                stdout="write",
                create_output_name="answer.txt",
            ),
        },
        services={
            "llm": llm,
            "sandbox": SimpleNamespace(),
            "workspace": workspace,
            "output_contract_config": OutputContractConfig(
                require_output_before_completion=True,
                missing_output_feedback_retries=1,
            ),
        },
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "out" / "answer.txt"

    await workflow.solve(
        description="Write the answer.",
        io_instructions="Write the answer to answer.txt.",
        data_dir=data_dir,
        output_path=output_path,
    )

    assert len(llm.calls) == 3
    assert (workspace.get_path("sandbox_workdir") / "answer.txt").exists()
    feedback_messages = [
        message["content"]
        for call in llm.calls
        for message in call
        if "Final answer rejected" in message["content"]
    ]
    assert feedback_messages
    messages_path = workspace.get_path("artifacts") / "messages.json"
    saved_messages = json.loads(messages_path.read_text(encoding="utf-8"))
    assert any("SubmissionStatus" in message["content"] for message in saved_messages)


@pytest.mark.asyncio
async def test_react_workflow_repairs_unclosed_answer_and_stops(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(["<Think>x</Think>\n<Answer>42"])

    workflow = ReActWorkflow(
        operators={"react": ReActOperator(max_steps=3), "execute": _FakeExecuteOperator(workspace)},
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "out" / "answer.txt"

    await workflow.solve(
        description="Return the final answer.",
        io_instructions="Write the answer to answer.txt.",
        data_dir=data_dir,
        output_path=output_path,
    )

    assert len(llm.calls) == 1
    messages_path = workspace.get_path("artifacts") / "messages.json"
    saved_messages = json.loads(messages_path.read_text(encoding="utf-8"))
    assert saved_messages[-1]["content"] == "<Think>x</Think>\n<Answer>42\n</Answer>"


@pytest.mark.asyncio
async def test_react_workflow_accepts_stripped_think_opening_and_executes(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            "Inspect first.</Think><Action>```python\nprint('run')\n```</Action>",
            "Done.</Think><Answer>finished</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(workspace, stdout="run")
    workflow = ReActWorkflow(
        operators={"react": ReActOperator(max_steps=2), "execute": execute_operator},
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    await workflow.solve(
        description="Inspect the data.",
        io_instructions="Return after the action succeeds.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    assert execute_operator.calls == [{"code": "print('run')", "mode": "script"}]
    saved_messages = json.loads(
        (workspace.get_path("artifacts") / "messages.json").read_text(encoding="utf-8")
    )
    assert saved_messages[2]["content"].startswith("Inspect first.</Think>")


@pytest.mark.asyncio
async def test_react_workflow_rejects_duplicate_actions_without_executing_and_recovers(
    tmp_path,
) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(
        [
            (
                "<Think>Try two actions.</Think>"
                "<Action>```python\nprint('must not run')\n```</Action>"
                "<Action>```python\nprint('must not run either')\n```</Action>"
            ),
            "<Think>Retry correctly.</Think><Action>```python\nprint('run once')\n```</Action>",
            "<Think>Done.</Think><Answer>finished</Answer>",
        ]
    )
    execute_operator = _FakeExecuteOperator(workspace, stdout="run once")
    workflow = ReActWorkflow(
        operators={"react": ReActOperator(max_steps=3), "execute": execute_operator},
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    await workflow.solve(
        description="Run exactly one valid action.",
        io_instructions="Return after the action succeeds.",
        data_dir=data_dir,
        output_path=tmp_path / "out" / "answer.txt",
    )

    assert execute_operator.calls == [{"code": "print('run once')", "mode": "script"}]
    assert len(llm.calls) == 3
    assert "Protocol error:" in llm.calls[1][-1]["content"]
    assert "exactly one <Action>" in llm.calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_react_workflow_leaves_directory_submission_artifact_in_sandbox(tmp_path) -> None:
    workspace = _DummyWorkspaceService(tmp_path / "workspace")
    llm = _DummyLLMService(["<Think>x</Think><Action>```python\nprint('dir')\n```</Action>"])

    workflow = ReActWorkflow(
        operators={
            "react": ReActOperator(max_steps=1),
            "execute": _FakeExecuteOperator(
                workspace,
                stdout="dir",
                create_directory_output_name="submission_bundle",
            ),
        },
        services={"llm": llm, "sandbox": SimpleNamespace(), "workspace": workspace},
        agent_config={},
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_path = tmp_path / "out" / "submission_bundle"

    await workflow.solve(
        description="Create a submission directory with the required covariance files.",
        io_instructions="Create submission_bundle in the working directory.",
        data_dir=data_dir,
        output_path=output_path,
    )

    sandbox_dir = workspace.get_path("sandbox_workdir") / "submission_bundle"
    assert sandbox_dir.is_dir()
    assert (sandbox_dir / "before_covariance.csv").exists()
    assert (sandbox_dir / "after_covariance.csv").exists()
    assert not output_path.exists()
