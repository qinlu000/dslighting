from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("diskcache")

from dslighting.benchmark.core.source_catalog import get_benchmark_source_catalog
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
from dslighting.config import DSLightingConfig, OutputContractConfig, TaskContextConfig
from dslighting.core.config import ConfigBuilder
from dslighting.core.task_context import (
    TaskContextBuilder,
    TaskContextError,
    load_l1_artifact,
)
from dslighting.core.task_context._markdown import replace_dataset_description
from dslighting.core.tasks import BaseTaskAdapter, FileSubmissionTaskAdapter
from dslighting.core.tasks.errors import TaskExecutionSpecError
from dslighting.core.tasks.models import ResolvedTaskLayout, TaskExecutionSpec
from dslighting.core.tasks.resolver import TaskResolver
from dslighting.core.types import TaskDefinition

TASK_ID = "dabench-test-task"
MOSCI_TASK_ID = "mosci-pop_genetics-3"
MOSCI_DATASET_ID = "mosci-pop_genetics"
ORIGINAL_DATA_DESCRIPTION = "ORIGINAL_DATA_DESCRIPTION_SENTINEL"
ORIGINAL_DATASET_DESCRIPTION = "ORIGINAL_DATASET_DESCRIPTION_SENTINEL"
GENERATED_L1_MEANING = "GENERATED_L1_SEMANTIC_MEANING_SENTINEL"
L2_GUIDANCE = "L2_GUIDANCE_SENTINEL"
DATA_PROFILE = "## Data Profile\n\nDATA_PROFILE_SENTINEL"
SUBMISSION_REQUIREMENTS = "## Submission Artifact Requirements\n\nSUBMISSION_REQUIREMENTS_SENTINEL"
DATA_REPORT = f"{DATA_PROFILE}\n\n{SUBMISSION_REQUIREMENTS}"
IO_INSTRUCTIONS = "IO_INSTRUCTIONS_SENTINEL"


class StubPerceptionRuntime:
    """Stable main-path collaborator; policy must not alter its output."""

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


class ExplicitIOPerceptionRuntime(StubPerceptionRuntime):
    """Prove a payload-owned I/O override short-circuits main generation."""

    def generate_io_instructions(
        self,
        output_filename: str,
        optimization_context: bool,
        submission_context: dict[str, object],
    ) -> str:
        del output_filename, optimization_context, submission_context
        raise AssertionError("explicit I/O must bypass perception generation")


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
    (public_dir / "train.csv").write_text("id,value\n1,10\n", encoding="utf-8")

    output_path = tmp_path / "submission.csv"
    submission = SubmissionArtifactContract(
        sample_submission_path=sample_path,
        output_submission_path=output_path,
        submission_filename="submission.csv",
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
    references = ReferenceArtifacts(
        task_root=task_root,
        raw_dir=raw_dir,
        public_dir=public_dir,
        private_dir=private_dir,
        answers_path=None,
        gold_submission_path=None,
        sample_submission_path=sample_path,
    )
    grading = TaskGradingContract(
        task_id=TASK_ID,
        source_id="dabench",
        engine_id="dabench",
        api_version="artifact_v1",
        grade_fn=lambda *args, **kwargs: 1.0,
        validate_fn=None,
        submission=submission,
        references=references,
        leaderboard_path=None,
    )
    evaluation_contract = TaskEvaluationContract(
        task_id=TASK_ID,
        source_id="dabench",
        engine_id="dabench",
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
        engine_id="dabench",
        evaluation_mode="artifact_submission",
        api_version="artifact_v1",
        registry_root=registry_root,
        data_root=tmp_path,
        mode="test",
    )
    description = f"""# Synthetic DABench Task

## Task Description

Compute the requested answer.

## Data Description

{ORIGINAL_DATA_DESCRIPTION}

### Nested Details

This nested subsection must be replaced with its parent section.

## Constraints

Keep this constraint unchanged.

## Submission Format

Keep the original submission instructions unchanged.
"""
    return ResolvedTaskLayout(
        task_id=TASK_ID,
        dataset_id=TASK_ID,
        source_id="dabench",
        engine_id="dabench",
        task_type="kaggle",
        registry_root=registry_root,
        task_root=task_root,
        data_root=tmp_path,
        agent_visible_dir=public_dir,
        description_text=description,
        sample_submission_path=sample_path,
        submission_filename="submission.csv",
        submission_format=".csv",
        submission_context=submission.to_payload(),
        output_path=output_path,
        evaluation_contract=evaluation_contract,
        evaluation_contract_ref=evaluation_ref,
    )


@pytest.fixture
def l1_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "l1"
    artifact_dir.mkdir()
    (artifact_dir / f"{TASK_ID}.json").write_text(
        json.dumps(
            {
                "schema_version": "l1_semantic_map_v1",
                "dataset_id": TASK_ID,
                "data_objects": [
                    {
                        "name": "training_table",
                        "files": ["train.csv"],
                        "kind": "table",
                        "meaning": GENERATED_L1_MEANING,
                    }
                ],
                "variables": [],
                "structure": [],
                "uncertainties": [],
                "annotation_notes": [],
            }
        ),
        encoding="utf-8",
    )
    return artifact_dir


@pytest.fixture
def mosci_layout(layout: ResolvedTaskLayout) -> ResolvedTaskLayout:
    grading = layout.evaluation_contract.grading
    assert grading is not None
    evaluation_contract = replace(
        layout.evaluation_contract,
        task_id=MOSCI_TASK_ID,
        source_id="moscibench",
        engine_id="mle",
        grading=replace(
            grading,
            task_id=MOSCI_TASK_ID,
            source_id="moscibench",
            engine_id="mle",
        ),
    )
    evaluation_ref = replace(
        layout.evaluation_contract_ref,
        task_id=MOSCI_TASK_ID,
        source_id="moscibench",
        engine_id="mle",
    )
    description = f"""# Synthetic MoSciBench Task

## Task

Discover the requested scientific relationship.

## Dataset Description

{ORIGINAL_DATASET_DESCRIPTION}

### Nested Dataset Details

This nested subsection must be replaced with its parent section.

## Submission Format

Keep the original submission instructions unchanged.
"""
    return replace(
        layout,
        task_id=MOSCI_TASK_ID,
        dataset_id=MOSCI_DATASET_ID,
        source_id="moscibench",
        engine_id="mle",
        description_text=description,
        evaluation_contract=evaluation_contract,
        evaluation_contract_ref=evaluation_ref,
    )


@pytest.fixture
def mosci_l1_artifact_dir(l1_artifact_dir: Path) -> Path:
    (l1_artifact_dir / f"{MOSCI_DATASET_ID}.json").write_text(
        json.dumps(
            {
                "schema_version": "l1_semantic_map_v1",
                "dataset_id": MOSCI_DATASET_ID,
                "data_objects": [
                    {
                        "name": "scientific_training_table",
                        "files": ["train.csv"],
                        "kind": "table",
                        "meaning": GENERATED_L1_MEANING,
                    }
                ],
                "variables": [],
                "structure": [],
                "uncertainties": [],
                "annotation_notes": [],
            }
        ),
        encoding="utf-8",
    )
    return l1_artifact_dir


@pytest.fixture
def l2_guidance_path(tmp_path: Path) -> Path:
    path = tmp_path / "l2.md"
    path.write_text(f"### Operating Strategy\n\n{L2_GUIDANCE}\n", encoding="utf-8")
    return path


def _builder(
    policy: str,
    *,
    l1_artifact_dir: Path | None = None,
    l2_guidance_path: Path | None = None,
) -> TaskContextBuilder:
    return TaskContextBuilder(
        TaskContextConfig(
            policy=policy,
            l1_artifact_dir=str(l1_artifact_dir) if l1_artifact_dir else None,
            l2_guidance_path=str(l2_guidance_path) if l2_guidance_path else None,
        ),
        StubPerceptionRuntime(),
    )


def _build_all_modes(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
) -> dict[str, TaskExecutionSpec]:
    return {
        "main": _builder("main").build(layout),
        "l1": _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout),
        "l2": _builder("l2", l2_guidance_path=l2_guidance_path).build(layout),
        "l3": _builder(
            "l3",
            l1_artifact_dir=l1_artifact_dir,
            l2_guidance_path=l2_guidance_path,
        ).build(layout),
    }


def test_main_matches_the_original_file_submission_contract(layout: ResolvedTaskLayout) -> None:
    runtime = StubPerceptionRuntime()
    actual = TaskContextBuilder(TaskContextConfig(), runtime).build(layout)

    assert actual.task_id == layout.task_id
    assert actual.task_type == layout.task_type
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
        TaskContextConfig(), runtime
    ).build(layout)


def test_l1_replaces_only_data_description(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
) -> None:
    main = _builder("main").build(layout)
    l1 = _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout)

    assert ORIGINAL_DATA_DESCRIPTION in main.description_text
    assert ORIGINAL_DATA_DESCRIPTION not in l1.description_text
    assert "This nested subsection must be replaced" not in l1.description_text
    assert l1.description_text.count("## Data Description") == 1
    assert l1.description_text.count(GENERATED_L1_MEANING) == 1
    assert "Compute the requested answer." in l1.description_text
    assert "Keep this constraint unchanged." in l1.description_text
    assert "Keep the original submission instructions unchanged." in l1.description_text
    assert L2_GUIDANCE not in l1.description_text


def test_l2_keeps_original_description_and_appends_guidance_once(
    layout: ResolvedTaskLayout,
    l2_guidance_path: Path,
) -> None:
    l2 = _builder("l2", l2_guidance_path=l2_guidance_path).build(layout)

    assert ORIGINAL_DATA_DESCRIPTION in l2.description_text
    assert l2.description_text.count("## Data Description") == 1
    assert GENERATED_L1_MEANING not in l2.description_text
    assert l2.description_text.count(L2_GUIDANCE) == 1
    assert l2.description_text.count("## Level 2 Guidance") == 1


def test_l3_combines_l1_replacement_and_l2_guidance_once(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
) -> None:
    l3 = _builder(
        "l3",
        l1_artifact_dir=l1_artifact_dir,
        l2_guidance_path=l2_guidance_path,
    ).build(layout)

    assert ORIGINAL_DATA_DESCRIPTION not in l3.description_text
    assert l3.description_text.count("## Data Description") == 1
    assert l3.description_text.count(GENERATED_L1_MEANING) == 1
    assert l3.description_text.count(L2_GUIDANCE) == 1


@pytest.mark.parametrize(
    ("policy", "uses_l1", "uses_l2"),
    [
        ("main", False, False),
        ("l1", True, False),
        ("l2", False, True),
        ("l3", True, True),
    ],
)
def test_moscibench_dataset_description_supports_every_policy(
    mosci_layout: ResolvedTaskLayout,
    mosci_l1_artifact_dir: Path,
    l2_guidance_path: Path,
    policy: str,
    uses_l1: bool,
    uses_l2: bool,
) -> None:
    spec = _builder(
        policy,
        l1_artifact_dir=mosci_l1_artifact_dir,
        l2_guidance_path=l2_guidance_path,
    ).build(mosci_layout)

    assert spec.description_text.count("## Dataset Description") == 1
    assert "Discover the requested scientific relationship." in spec.description_text
    assert "Keep the original submission instructions unchanged." in spec.description_text
    assert spec.description_text.count(DATA_REPORT) == 1
    assert spec.io_instructions == IO_INSTRUCTIONS

    if uses_l1:
        assert ORIGINAL_DATASET_DESCRIPTION not in spec.description_text
        assert "This nested subsection must be replaced" not in spec.description_text
        assert spec.description_text.count(GENERATED_L1_MEANING) == 1
    else:
        assert ORIGINAL_DATASET_DESCRIPTION in spec.description_text
        assert "This nested subsection must be replaced" in spec.description_text
        assert GENERATED_L1_MEANING not in spec.description_text

    if uses_l2:
        assert spec.description_text.count("## Level 2 Guidance") == 1
        assert spec.description_text.count(L2_GUIDANCE) == 1
    else:
        assert "## Level 2 Guidance" not in spec.description_text
        assert L2_GUIDANCE not in spec.description_text


def test_all_policies_preserve_data_profile_submission_requirements_and_io(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
) -> None:
    specs = _build_all_modes(layout, l1_artifact_dir, l2_guidance_path)
    main = specs["main"]
    expected_report_hash = hashlib.sha256(DATA_REPORT.encode("utf-8")).hexdigest()
    expected_l1_path = (l1_artifact_dir / f"{TASK_ID}.json").resolve()
    expected_l1_hash = hashlib.sha256(expected_l1_path.read_bytes()).hexdigest()
    expected_l2_path = l2_guidance_path.resolve()
    expected_l2_hash = hashlib.sha256(expected_l2_path.read_bytes()).hexdigest()

    for policy, spec in specs.items():
        assert spec.description_text.count(DATA_REPORT) == 1
        assert spec.description_text.count(DATA_PROFILE) == 1
        assert spec.description_text.count(SUBMISSION_REQUIREMENTS) == 1
        assert spec.io_instructions == IO_INSTRUCTIONS
        assert spec.output_path == main.output_path
        assert spec.submission_artifact_contract == main.submission_artifact_contract
        assert spec.evaluation_contract_ref == main.evaluation_contract_ref

        provenance = spec.task_context_provenance
        assert provenance is not None
        assert provenance["policy"] == policy
        assert provenance["main_data_report_sha256"] == expected_report_hash

        if policy in {"l1", "l3"}:
            assert Path(provenance["l1"]["path"]).resolve() == expected_l1_path
            assert provenance["l1"]["sha256"] == expected_l1_hash
        else:
            assert provenance["l1"] is None

        if policy in {"l2", "l3"}:
            assert Path(provenance["l2"]["path"]).resolve() == expected_l2_path
            assert provenance["l2"]["sha256"] == expected_l2_hash
        else:
            assert provenance["l2"] is None


def test_real_main_perception_is_fixed_across_policy_log_directories(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
    tmp_path: Path,
) -> None:
    from dslighting.services.data_analysis_provider import (
        create_data_perception_runtime,
    )

    runtime = create_data_perception_runtime(
        DSLightingConfig.model_validate(
            {"data_analysis": {"cache_enabled": False, "max_report_chars": 14000}}
        )
    )
    assert runtime is not None

    specs: dict[str, TaskExecutionSpec] = {}
    for policy in ("main", "l1", "l2", "l3"):
        output_path = tmp_path / policy / "submission-stable.csv"
        grading = layout.evaluation_contract.grading
        assert grading is not None
        submission = grading.submission.with_output_path(output_path)
        policy_layout = replace(
            layout,
            output_path=output_path,
            submission_filename=output_path.name,
            submission_context=submission.to_payload(),
            evaluation_contract=replace(
                layout.evaluation_contract,
                grading=replace(grading, submission=submission),
            ),
        )
        specs[policy] = TaskContextBuilder(
            TaskContextConfig(
                policy=policy,
                l1_artifact_dir=str(l1_artifact_dir),
                l2_guidance_path=str(l2_guidance_path),
            ),
            runtime,
        ).build(policy_layout)

    report_hashes = {
        spec.task_context_provenance["main_data_report_sha256"]
        for spec in specs.values()
        if spec.task_context_provenance is not None
    }
    io_hashes = {
        hashlib.sha256(spec.io_instructions.encode("utf-8")).hexdigest() for spec in specs.values()
    }
    assert len(report_hashes) == 1
    assert len(io_hashes) == 1


@pytest.mark.parametrize(
    ("policy", "l1_artifact_dir", "l2_guidance_path"),
    [
        ("l1", None, None),
        ("l2", None, None),
        ("l3", None, None),
    ],
)
def test_enrichment_policies_fail_closed_when_required_inputs_are_missing(
    layout: ResolvedTaskLayout,
    policy: str,
    l1_artifact_dir: Path | None,
    l2_guidance_path: Path | None,
) -> None:
    with pytest.raises(TaskContextError):
        _builder(
            policy,
            l1_artifact_dir=l1_artifact_dir,
            l2_guidance_path=l2_guidance_path,
        ).build(layout)


def test_l1_fails_closed_when_task_artifact_is_absent(
    layout: ResolvedTaskLayout,
    tmp_path: Path,
) -> None:
    empty_artifact_dir = tmp_path / "empty-l1"
    empty_artifact_dir.mkdir()

    with pytest.raises(TaskContextError):
        _builder("l1", l1_artifact_dir=empty_artifact_dir).build(layout)


def test_l1_requires_an_explicit_schema_version(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
) -> None:
    artifact_path = l1_artifact_dir / f"{layout.task_id}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact.pop("schema_version")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(TaskContextError):
        _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout)


def test_l1_render_is_stable_when_structure_object_order_changes(tmp_path: Path) -> None:
    base_payload = {
        "schema_version": "l1_semantic_map_v1",
        "dataset_id": TASK_ID,
        "data_objects": [
            {
                "name": name,
                "files": [f"{name}.csv"],
                "kind": "table",
                "meaning": f"Public {name} records.",
            }
            for name in ("alpha", "beta", "gamma")
        ],
        "variables": [],
        "structure": [
            {"description": "Related records.", "objects": ["beta", "alpha"]},
            {"description": "Related records.", "objects": ["alpha", "gamma"]},
        ],
        "uncertainties": [],
        "annotation_notes": [],
    }
    rendered: list[str] = []
    for directory_name, object_order in (
        ("first", ["beta", "alpha"]),
        ("second", ["alpha", "beta"]),
    ):
        artifact_dir = tmp_path / directory_name
        artifact_dir.mkdir()
        payload = json.loads(json.dumps(base_payload))
        payload["structure"][0]["objects"] = object_order
        (artifact_dir / f"{TASK_ID}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        rendered.append(load_l1_artifact(artifact_dir, TASK_ID).render_markdown())

    assert rendered[0] == rendered[1]


def test_l1_cannot_describe_the_configured_output_as_public_input(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
) -> None:
    public_output = layout.agent_visible_dir / "solver-output.csv"
    public_output.write_text("answer\nplaceholder\n", encoding="utf-8")
    artifact_path = l1_artifact_dir / f"{layout.task_id}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["data_objects"][0]["files"] = [public_output.name]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(TaskContextError):
        _builder("l1", l1_artifact_dir=l1_artifact_dir).build(
            replace(layout, output_path=public_output)
        )


def test_l1_must_describe_every_visible_public_input(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
) -> None:
    (layout.agent_visible_dir / "uncovered.csv").write_text(
        "value\n2\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskContextError, match="undescribed"):
        _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout)


def test_l1_rejects_visible_directory_symlink_that_escapes_public_root(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    tmp_path: Path,
) -> None:
    private_directory = tmp_path / "private-data"
    private_directory.mkdir()
    (private_directory / "answers.csv").write_text("answer\n42\n", encoding="utf-8")
    (layout.agent_visible_dir / "linked-private").symlink_to(
        private_directory,
        target_is_directory=True,
    )

    with pytest.raises(TaskContextError, match="escapes the public root"):
        _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout)


def test_l1_rejects_private_or_grader_content_hidden_in_semantic_prose(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
) -> None:
    artifact_path = l1_artifact_dir / f"{layout.task_id}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["data_objects"][0]["meaning"] = "The private grader reveals the task answer."
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(TaskContextError, match="disallowed content"):
        _builder("l1", l1_artifact_dir=l1_artifact_dir).build(layout)


def test_l3_fails_closed_when_either_enrichment_input_is_missing(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
) -> None:
    with pytest.raises(TaskContextError):
        _builder("l3", l1_artifact_dir=l1_artifact_dir).build(layout)

    with pytest.raises(TaskContextError):
        _builder("l3", l2_guidance_path=l2_guidance_path).build(layout)


@pytest.mark.parametrize(
    "forbidden_title",
    [
        "Output Contract",
        "Output Contract Details",
        "I/O Instructions",
        "Task Answer",
        "Private Data",
        "Grader Facts",
    ],
)
def test_l2_cannot_redeclare_owned_or_disallowed_sections(
    layout: ResolvedTaskLayout,
    tmp_path: Path,
    forbidden_title: str,
) -> None:
    guidance_path = tmp_path / "invalid-l2.md"
    guidance_path.write_text(
        f"### {forbidden_title}\n\nDo something different.\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskContextError):
        _builder("l2", l2_guidance_path=guidance_path).build(layout)


@pytest.mark.parametrize(
    "structural_heading",
    [
        "## A New Top-Level Section",
        "   ## An Indented Top-Level Section",
        "Setext Level Two\n----------------",
        "Setext Level One\n================",
        "<h2>Raw HTML Level Two</h2>",
        '<h2\n class="escaped-section">Multiline Raw HTML Level Two</h2>',
    ],
)
def test_l2_artifact_cannot_escape_its_runtime_owned_section(
    layout: ResolvedTaskLayout,
    tmp_path: Path,
    structural_heading: str,
) -> None:
    guidance_path = tmp_path / "invalid-l2.md"
    guidance_path.write_text(
        f"{structural_heading}\n\nThis would escape the L2 wrapper.\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskContextError):
        _builder("l2", l2_guidance_path=guidance_path).build(layout)


def test_l2_structural_heading_syntax_inside_fenced_code_is_not_a_section(
    layout: ResolvedTaskLayout,
    tmp_path: Path,
) -> None:
    guidance_path = tmp_path / "code-example-l2.md"
    guidance_path.write_text(
        "### Operating Strategy\n\n"
        "```markdown\n"
        "Setext Example\n"
        "--------------\n"
        "   ## Indented Example\n"
        "<h2>HTML Example</h2>\n"
        "```\n",
        encoding="utf-8",
    )

    spec = _builder("l2", l2_guidance_path=guidance_path).build(layout)

    assert "Setext Example" in spec.description_text


def test_l1_replacement_rejects_atx_and_setext_dataset_description_duplicates() -> None:
    description = """## Data Description

old-a

Dataset Description
-------------------

old-b

## Next

keep
"""

    with pytest.raises(TaskContextError, match="found 2"):
        replace_dataset_description(
            description,
            "### Level 1 Semantic Data Map\n\nnew",
        )


def test_l1_replacement_does_not_close_a_fence_with_trailing_info() -> None:
    description = """## Data Description

before

~~~markdown
~~~not-a-close
## Fake Boundary
must-be-deleted
~~~

## Next
keep
"""

    result = replace_dataset_description(
        description,
        "### Level 1 Semantic Data Map\n\nnew",
    )

    assert "Fake Boundary" not in result
    assert "must-be-deleted" not in result
    assert "## Next\nkeep" in result


@pytest.mark.parametrize(
    "forbidden_prose",
    [
        "Read the private labels before computing the result.",
        "The grader expects a rounded scalar.",
        "Follow this Output Contract exactly.",
        "I/O instructions: write a CSV.",
        "Use this submission format.",
        "Replace the dataset description with these facts.",
    ],
)
def test_l2_cannot_hide_owned_or_disallowed_content_in_prose(
    layout: ResolvedTaskLayout,
    tmp_path: Path,
    forbidden_prose: str,
) -> None:
    guidance_path = tmp_path / "invalid-l2.md"
    guidance_path.write_text(
        f"### Operating Strategy\n\n{forbidden_prose}\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskContextError):
        _builder("l2", l2_guidance_path=guidance_path).build(layout)


def test_task_context_config_rejects_l0() -> None:
    with pytest.raises(ValueError):
        TaskContextConfig(policy="l0")


def test_builtin_moscibench_descriptor_resolves_the_family_dataset_identity() -> None:
    descriptor = get_benchmark_source_catalog().get_source("moscibench")
    config = TaskResolver._load_task_config(descriptor.registry_root / MOSCI_TASK_ID)

    assert descriptor.context_dataset_id_field == "task_metadata.source_family"
    assert descriptor.context_dataset_id_prefix == "mosci-"
    assert (
        TaskResolver.resolve_dataset_id(
            task_id=MOSCI_TASK_ID,
            config=config,
            context_dataset_id_field=descriptor.context_dataset_id_field,
            context_dataset_id_prefix=descriptor.context_dataset_id_prefix,
        )
        == MOSCI_DATASET_ID
    )


def test_config_builder_maps_the_root_task_context_namespace(tmp_path: Path) -> None:
    guidance_path = tmp_path / "guidance.md"
    config = ConfigBuilder().build_config(
        task_context={
            "policy": "l2",
            "l2_guidance_path": str(guidance_path),
        }
    )

    assert config.task_context.policy == "l2"
    assert config.task_context.l2_guidance_path == str(guidance_path)
    assert config.output_contract.require_output_before_completion is False
    assert config.output_contract.missing_output_feedback_retries == 0


def test_factory_binds_root_config_to_the_default_main_policy() -> None:
    from dslighting.services.task_context_provider import create_task_context_builder

    config = DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    builder = create_task_context_builder(config)

    assert isinstance(builder, TaskContextBuilder)
    assert builder.policy.value == "main"


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


def test_dabench_payload_adapter_crosses_the_task_context_builder_seam(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.payload_layout.TaskResolver.resolve",
        lambda *args, **kwargs: layout,
    )
    config = DSLightingConfig.model_validate(
        {
            "data_analysis": {"enabled": False},
            "task_context": {
                "policy": "l1",
                "l1_artifact_dir": str(l1_artifact_dir),
            },
        }
    )
    adapter = FileSubmissionTaskAdapter(config)
    payload_output = layout.output_path.with_name("benchmark-output.csv")
    submission = layout.evaluation_contract.grading.submission.with_output_path(payload_output)
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(payload_output),
            **submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert ORIGINAL_DATA_DESCRIPTION not in spec.description_text
    assert GENERATED_L1_MEANING in spec.description_text
    assert spec.task_context_provenance is not None
    assert spec.task_context_provenance["policy"] == "l1"
    assert spec.output_path == payload_output
    assert spec.submission_artifact_contract is not None
    assert spec.submission_artifact_contract.output_submission_path == payload_output
    assert spec.submission_artifact_contract.validation.expected_name == payload_output.name


def test_main_payload_bridge_preserves_disabled_perception_io_bytes(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.payload_layout.TaskResolver.resolve",
        lambda *args, **kwargs: layout,
    )
    config = DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    adapter = FileSubmissionTaskAdapter(config)
    submission = layout.evaluation_contract.grading.submission
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
            **submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert spec.io_instructions == (
        "All input data files are in the current working directory.\n"
        f"Save the final submission artifact to `{layout.output_path.name}` in the "
        "current working directory."
    )


def test_payload_bridge_preserves_explicit_io_instructions(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.payload_layout.TaskResolver.resolve",
        lambda *args, **kwargs: layout,
    )
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
            "io_instructions": "EXPLICIT_IO_SENTINEL",
            **layout.evaluation_contract.grading.submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert spec.io_instructions == "EXPLICIT_IO_SENTINEL"


def test_payload_explicit_io_short_circuits_perception_generation(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.payload_layout.TaskResolver.resolve",
        lambda *args, **kwargs: layout,
    )
    adapter = FileSubmissionTaskAdapter(
        DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    )
    runtime = ExplicitIOPerceptionRuntime()
    adapter.data_perception = runtime
    adapter.task_context_builder = TaskContextBuilder(TaskContextConfig(), runtime)
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
            "io_instructions": "  EXPLICIT_IO_SENTINEL  ",
            **layout.evaluation_contract.grading.submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert spec.io_instructions == "EXPLICIT_IO_SENTINEL"


def test_main_payload_bridge_preserves_explicit_metric_semantics(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.payload_layout.TaskResolver.resolve",
        lambda *args, **kwargs: layout,
    )
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

    assert spec.metric_name == "custom_metric"
    assert spec.lower_is_better is True
    assert spec.source_id == "explicit-source"
    assert spec.engine_id == "explicit-engine"
    assert spec.evaluation_contract_ref == layout.evaluation_contract_ref


@pytest.mark.parametrize(
    ("policy", "uses_l1", "uses_l2"),
    [
        ("main", False, False),
        ("l1", True, False),
        ("l2", False, True),
        ("l3", True, True),
    ],
)
def test_ablation_payload_arms_share_the_canonical_layout_seam(
    mosci_layout: ResolvedTaskLayout,
    mosci_l1_artifact_dir: Path,
    l2_guidance_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    uses_l1: bool,
    uses_l2: bool,
) -> None:
    layout = mosci_layout
    resolution_calls: list[dict[str, object]] = []

    def resolve_canonical_layout(**kwargs: object) -> ResolvedTaskLayout:
        resolution_calls.append(kwargs)
        return layout

    monkeypatch.setattr(
        "dslighting.core.tasks.adapters.resolve_file_submission_payload_layout",
        resolve_canonical_layout,
    )
    config = DSLightingConfig.model_validate(
        {
            "data_analysis": {"enabled": False},
            "task_context": {
                "policy": policy,
                "l1_artifact_dir": str(mosci_l1_artifact_dir),
                "l2_guidance_path": str(l2_guidance_path),
                "require_canonical_layout": True,
            },
        }
    )
    adapter = FileSubmissionTaskAdapter(config)
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "registry_dir": str(layout.registry_root),
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
            **layout.evaluation_contract.grading.submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert len(resolution_calls) == 1
    assert resolution_calls[0]["task_id"] == layout.task_id
    assert resolution_calls[0]["registry_dir"] == str(layout.registry_root)
    assert resolution_calls[0]["data_dir"] == layout.agent_visible_dir
    assert resolution_calls[0]["output_path"] == layout.output_path
    assert spec.task_context_provenance is not None
    assert spec.task_context_provenance["policy"] == policy
    assert spec.output_path == layout.output_path

    if uses_l1:
        assert ORIGINAL_DATASET_DESCRIPTION not in spec.description_text
        assert GENERATED_L1_MEANING in spec.description_text
        assert Path(spec.task_context_provenance["l1"]["path"]).name == f"{MOSCI_DATASET_ID}.json"
    else:
        assert ORIGINAL_DATASET_DESCRIPTION in spec.description_text
        assert GENERATED_L1_MEANING not in spec.description_text

    if uses_l2:
        assert spec.description_text.count(L2_GUIDANCE) == 1
    else:
        assert L2_GUIDANCE not in spec.description_text


def test_ordinary_main_payload_preserves_the_legacy_path_without_resolution(
    layout: ResolvedTaskLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_resolution(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("ordinary main must not enter the canonical layout seam")

    monkeypatch.setattr(
        "dslighting.core.tasks.adapters.resolve_file_submission_payload_layout",
        unexpected_resolution,
    )
    config = DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    adapter = FileSubmissionTaskAdapter(config)
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "registry_dir": str(layout.registry_root),
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
            **layout.evaluation_contract.grading.submission.to_payload(),
        },
    )

    spec = adapter.build_execution_spec(task)
    adapter.cleanup()

    assert spec.description_text == layout.description_text.strip()
    assert spec.output_path == layout.output_path
    assert spec.task_context_provenance is None
    assert spec.io_instructions == (
        "All input data files are in the current working directory.\n"
        f"Save the final submission artifact to `{layout.output_path.name}` in the "
        "current working directory."
    )


def test_treatment_payload_never_falls_back_when_layout_is_unavailable(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dslighting.core.tasks.adapters.resolve_file_submission_payload_layout",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no registry")),
    )
    config = DSLightingConfig.model_validate(
        {
            "data_analysis": {"enabled": False},
            "task_context": {
                "policy": "l1",
                "l1_artifact_dir": str(l1_artifact_dir),
            },
        }
    )
    adapter = FileSubmissionTaskAdapter(config)
    task = TaskDefinition(
        task_id=layout.task_id,
        task_type="kaggle",
        payload={
            "description": layout.description_text,
            "agent_visible_data_dir": str(layout.agent_visible_dir),
            "output_submission_path": str(layout.output_path),
        },
    )

    with pytest.raises(TaskExecutionSpecError, match="canonical task layout"):
        adapter.build_execution_spec(task)
    adapter.cleanup()


def test_execution_spec_round_trip_preserves_task_context_provenance(
    layout: ResolvedTaskLayout,
    l1_artifact_dir: Path,
    l2_guidance_path: Path,
) -> None:
    original = _builder(
        "l3",
        l1_artifact_dir=l1_artifact_dir,
        l2_guidance_path=l2_guidance_path,
    ).build(layout)
    adapter = FileSubmissionTaskAdapter(
        DSLightingConfig.model_validate({"data_analysis": {"enabled": False}})
    )

    restored = adapter.build_execution_spec(
        TaskDefinition(
            task_id=layout.task_id,
            task_type=layout.task_type,
            payload=original.to_payload(),
        )
    )
    adapter.cleanup()

    assert restored == original
    assert restored.task_context_provenance == original.task_context_provenance


def test_output_contract_defaults_remain_opt_in() -> None:
    config = OutputContractConfig()

    assert config.require_output_before_completion is False
    assert config.missing_output_feedback_retries == 0
