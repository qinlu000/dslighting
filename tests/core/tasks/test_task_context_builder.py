from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("diskcache")

from dslighting.benchmark.evaluation.models import (
    EvaluationSemantics,
    TaskEvaluationContract,
    TaskEvaluationContractRef,
)
from dslighting.benchmark.grading.models import (
    ReferenceArtifacts,
    SubmissionArtifactContract,
    SubmissionEntrySpec,
    SubmissionValidationSpec,
    TaskGradingContract,
)
from dslighting.config import DSLightingConfig, OutputContractConfig
from dslighting.core.task_context import TaskContextBuilder
from dslighting.core.tasks import BaseTaskAdapter, FileSubmissionTaskAdapter
from dslighting.core.tasks.models import ResolvedTaskLayout
from dslighting.core.types import TaskDefinition

TASK_ID = "dabench-test-task"
DATA_REPORT = "## Data Profile\n\nDATA_PROFILE_SENTINEL"
IO_INSTRUCTIONS = "IO_INSTRUCTIONS_SENTINEL"


class StubPerceptionRuntime:
    def analyze_data(
        self,
        data_dir: Path,
        task_type: str,
        task_id: str,
        submission_context: dict[str, object],
    ) -> str:
        del data_dir, task_type, task_id, submission_context
        return DATA_REPORT

    def generate_io_instructions(
        self,
        output_filename: str,
        optimization_context: bool,
        submission_context: dict[str, object],
    ) -> str:
        del output_filename, optimization_context, submission_context
        return IO_INSTRUCTIONS


@pytest.fixture
def layout(tmp_path: Path) -> ResolvedTaskLayout:
    registry_root = tmp_path / "registry"
    task_root = registry_root / TASK_ID
    public_dir = task_root / "prepared" / "public"
    private_dir = task_root / "prepared" / "private"
    raw_dir = task_root / "raw"
    for directory in (public_dir, private_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample_path = public_dir / "sample_submission.csv"
    sample_path.write_text("id,answer\n1,placeholder\n", encoding="utf-8")
    output_path = tmp_path / "submission.csv"
    submission = SubmissionArtifactContract(
        sample_submission_path=sample_path,
        output_submission_path=output_path,
        submission_filename=output_path.name,
        submission_format=".csv",
        validation=SubmissionValidationSpec(
            expected_kind="file",
            expected_name=output_path.name,
            allowed_suffixes=(".csv",),
        ),
        entries=(
            SubmissionEntrySpec(
                relative_path=output_path.name,
                format="csv",
                sample_path=sample_path,
            ),
        ),
    )
    grading = TaskGradingContract(
        task_id=TASK_ID,
        source_id="dabench",
        engine_id="mle",
        api_version="artifact_v1",
        grade_fn=lambda *args, **kwargs: 1.0,
        validate_fn=None,
        submission=submission,
        references=ReferenceArtifacts(
            task_root=task_root,
            raw_dir=raw_dir,
            public_dir=public_dir,
            private_dir=private_dir,
            answers_path=None,
            gold_submission_path=None,
            sample_submission_path=sample_path,
        ),
        leaderboard_path=None,
    )
    evaluation_contract = TaskEvaluationContract(
        task_id=TASK_ID,
        source_id="dabench",
        engine_id="mle",
        evaluation_mode="artifact_submission",
        api_version="artifact_v1",
        evaluation_semantics=EvaluationSemantics(
            objective="higher_is_better",
            leaderboard_path=None,
        ),
        grading=grading,
        judging=None,
    )
    evaluation_ref = TaskEvaluationContractRef(
        task_id=TASK_ID,
        source_id="dabench",
        engine_id="mle",
        evaluation_mode="artifact_submission",
        api_version="artifact_v1",
        registry_root=registry_root,
        data_root=tmp_path,
        mode="test",
    )
    return ResolvedTaskLayout(
        task_id=TASK_ID,
        dataset_id=TASK_ID,
        source_id="dabench",
        engine_id="mle",
        task_type="kaggle",
        registry_root=registry_root,
        task_root=task_root,
        data_root=tmp_path,
        agent_visible_dir=public_dir,
        description_text="Solve the task.",
        sample_submission_path=sample_path,
        submission_filename=output_path.name,
        submission_format=".csv",
        submission_context=submission.to_payload(),
        output_path=output_path,
        evaluation_contract=evaluation_contract,
        evaluation_contract_ref=evaluation_ref,
    )


def test_builder_preserves_the_main_file_submission_contract(
    layout: ResolvedTaskLayout,
) -> None:
    actual = TaskContextBuilder(StubPerceptionRuntime()).build(layout)

    assert actual.description_text == f"{layout.description_text}\n\n{DATA_REPORT}"
    assert actual.io_instructions == IO_INSTRUCTIONS
    assert actual.agent_visible_dir == layout.agent_visible_dir
    assert actual.output_path == layout.output_path
    assert actual.metric_name == "score"
    assert actual.lower_is_better is False
    assert actual.source_id == layout.source_id
    assert actual.engine_id == layout.engine_id
    assert actual.submission_artifact_contract == (
        layout.evaluation_contract.grading.submission.with_output_path(layout.output_path)
    )
    assert actual.evaluation_contract_ref == layout.evaluation_contract_ref


def test_legacy_file_submission_helper_routes_through_the_builder(
    layout: ResolvedTaskLayout,
) -> None:
    runtime = StubPerceptionRuntime()

    assert BaseTaskAdapter.build_file_submission_spec(layout, runtime) == TaskContextBuilder(
        runtime
    ).build(layout)


def test_factory_uses_an_explicit_perception_runtime(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dslighting.services import task_context_provider

    def unexpected_runtime_creation(config: DSLightingConfig) -> None:
        del config
        raise AssertionError("explicit perception runtime must bypass runtime creation")

    monkeypatch.setattr(
        task_context_provider,
        "create_data_perception_runtime",
        unexpected_runtime_creation,
    )
    builder = task_context_provider.create_task_context_builder(
        DSLightingConfig(),
        perception_runtime=StubPerceptionRuntime(),
    )

    spec = builder.build(layout)
    assert DATA_REPORT in spec.description_text
    assert spec.io_instructions == IO_INSTRUCTIONS


def test_payload_adapter_preserves_explicit_io_and_metric_semantics(
    layout: ResolvedTaskLayout,
) -> None:
    adapter = FileSubmissionTaskAdapter(
        DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    )
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
            "io_instructions": "  EXPLICIT_IO_SENTINEL  ",
            "metric_name": "custom_metric",
            "lower_is_better": True,
            "source_id": "explicit-source",
            "engine_id": "explicit-engine",
            **layout.evaluation_contract_ref.to_payload(),
            **layout.evaluation_contract.grading.submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert spec.io_instructions == "EXPLICIT_IO_SENTINEL"
    assert spec.metric_name == "custom_metric"
    assert spec.lower_is_better is True
    assert spec.source_id == "explicit-source"
    assert spec.engine_id == "explicit-engine"
    assert spec.evaluation_contract_ref == layout.evaluation_contract_ref


def test_output_contract_defaults_remain_opt_in() -> None:
    config = OutputContractConfig()

    assert config.require_output_before_completion is False
    assert config.missing_output_feedback_retries == 0
