from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from dslighting.config import DSLightingConfig
from dslighting.core.tasks import TaskExecutionSpec
from dslighting.core.types import TaskDefinition
from dslighting.runner import DSLightingRunner


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.writes: list[tuple[str, str, str]] = []

    def write_file(self, content: str, area: str, path: str) -> None:
        self.writes.append((content, area, path))

    def get_path(self, area: str) -> Path:
        return self.root / area


def _execution_spec(tmp_path: Path) -> TaskExecutionSpec:
    return TaskExecutionSpec(
        task_id="dabench-task",
        task_type="kaggle",
        description_text="description",
        io_instructions="explicit final I/O",
        agent_visible_dir=tmp_path,
        output_path=tmp_path / "submission.csv",
        task_context_provenance={
            "policy": "main",
            "main_data_report_sha256": "report-hash",
            "l1": None,
            "l2": None,
        },
    )


def test_fixed_context_uses_final_solver_io_and_output_basename(tmp_path: Path) -> None:
    audit = DSLightingRunner._build_task_context_audit(_execution_spec(tmp_path))

    assert audit["fixed_context"] == {
        "io_instructions_sha256": hashlib.sha256(b"explicit final I/O").hexdigest(),
        "output_artifact_name": "submission.csv",
        "submission_contract_sha256": hashlib.sha256(b"null").hexdigest(),
        "evaluation_contract_ref_sha256": hashlib.sha256(b"null").hexdigest(),
    }


@pytest.mark.asyncio
async def test_execution_spec_is_observed_before_workflow_failure(tmp_path: Path) -> None:
    spec = _execution_spec(tmp_path)

    class Adapter:
        def __init__(self, config: DSLightingConfig) -> None:
            del config

        def build_execution_spec(self, task: TaskDefinition) -> TaskExecutionSpec:
            del task
            return spec

    async def fail_workflow(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("workflow failed after spec construction")

    runner = object.__new__(DSLightingRunner)
    runner.adapter_classes = {"kaggle": Adapter}
    runner._execute_workflow_entrypoint = fail_workflow
    observed: list[TaskExecutionSpec] = []
    task = TaskDefinition(task_id="dabench-task", task_type="kaggle", payload={})

    with pytest.raises(RuntimeError, match="after spec construction"):
        await runner._execute_task_adapter(
            task,
            workflow=object(),
            llm_service=None,
            sandbox_service=None,
            workspace_service=None,
            task_config=DSLightingConfig(),
            execution_spec_observer=observed.append,
        )

    assert observed == [spec]


def test_run_record_keeps_task_context_provenance(tmp_path: Path) -> None:
    runner = object.__new__(DSLightingRunner)
    runner.run_records = []
    provenance = {
        "policy": "l1",
        "main_data_report_sha256": "report-hash",
        "l1": {"path": "/l1.json", "sha256": "l1-hash"},
        "l2": None,
    }
    fixed_context = {
        "io_instructions_sha256": "io-hash",
        "output_artifact_name": "submission.csv",
    }
    task = TaskDefinition(task_id="dabench-task", task_type="kaggle", payload={})
    now = datetime.utcnow()
    metadata = runner._build_metadata_dict(
        task_config=DSLightingConfig(),
        task=task,
        description="description",
        io_instructions="instructions",
        task_context_audit={
            "provenance": provenance,
            "fixed_context": fixed_context,
        },
        data_dir=tmp_path,
        output_path=tmp_path / "submission.csv",
        result=None,
        llm_calls=[],
        sandbox_runs=[],
        best_node=None,
        search_tree_info=None,
        dag_summary=None,
        started_at=now,
        ended_at=now,
        duration_seconds=0.0,
        total_cost=0.0,
        run_context={
            "workspace_dir": None,
            "filtered_parameters": {},
            "benchmark_snapshot": {},
            "usage_summary": {},
            "final_code_path": None,
            "config_snapshot": {},
        },
    )

    workspace = _Workspace(tmp_path)
    runner._persist_metadata_and_record(
        workspace_service=workspace,
        task=task,
        metadata=metadata,
        telemetry_dir="telemetry",
    )

    assert metadata["task_context"]["audit"]["provenance"] == provenance
    assert runner.get_run_records()[0]["task_context_audit"] == {
        "provenance": provenance,
        "fixed_context": fixed_context,
    }
