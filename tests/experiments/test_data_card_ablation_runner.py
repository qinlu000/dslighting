from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import dslighting.core.task_context as task_context_boundary
from dslighting.benchmark.core.source_catalog import get_benchmark_source_catalog
from dslighting.core.task_context._markdown import replace_dataset_description
from experiments.data_card_ablation import engine
from experiments.data_card_ablation import run_ablation as runner
from experiments.data_card_ablation.benchmark_profiles import get_ablation_profile

TASK_ID = "dabench-test-task"
DABENCH_TASK_ID = "dabench-0-mean-fare-paid"
MOSCIBENCH_TASK_ID = "mosci-cyclone-1"
TARGET_DESCRIPTION = """# Task

## Data Description

Original public data description.

## Submission Format

Write the required artifact.
"""


def _target(
    *,
    task_id: str,
    dataset_id: str,
    public_dir: Path,
    sample_submission_path: Path | None,
) -> engine.TaskTarget:
    return engine.TaskTarget(
        task_id=task_id,
        dataset_id=dataset_id,
        public_dir=public_dir,
        sample_submission_path=sample_submission_path,
        description_text=TARGET_DESCRIPTION,
        description_sha256=engine.sha256(TARGET_DESCRIPTION.encode("utf-8")),
    )


def _benchmark_profile(source_id: str) -> tuple[Any, Any, Any]:
    catalog = get_benchmark_source_catalog()
    descriptor = catalog.get_source(source_id)
    return catalog, descriptor, get_ablation_profile(descriptor.source_id)


def _write_l1_artifact(
    artifact_dir: Path,
    referenced_file: str,
    *,
    artifact_id: str = TASK_ID,
    dataset_id: str = TASK_ID,
    task_id: str | None = TASK_ID,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{artifact_id}.json"
    identity = {"dataset_id": dataset_id}
    if task_id is not None:
        identity["task_id"] = task_id
    path.write_text(
        json.dumps(
            {
                "schema_version": "l1_semantic_map_v1",
                **identity,
                "data_objects": [
                    {
                        "name": "public_table",
                        "files": [referenced_file],
                        "kind": "table",
                        "meaning": "The public records used by this dataset.",
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
    return path


def _make_task_roots(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    public_dir = data_root / TASK_ID / "prepared" / "public"
    public_dir.mkdir(parents=True)
    (public_dir / "train.csv").write_text("value\n1\n", encoding="utf-8")
    (public_dir / "sample_submission.csv").write_text(
        "output\n0\n",
        encoding="utf-8",
    )

    return data_root, public_dir


def _make_prepared_source_task(
    tmp_path: Path,
    *,
    task_id: str,
    nested_competitions: bool = False,
) -> tuple[Path, Path, Path]:
    requested_root = tmp_path / "release"
    data_root = requested_root / "competitions" if nested_competitions else requested_root
    public_dir = data_root / task_id / "prepared" / "public"
    public_dir.mkdir(parents=True)
    (public_dir / "train.csv").write_text("value\n1\n", encoding="utf-8")
    sample_submission = public_dir / "sample_submission.csv"
    sample_submission.write_text("output\n0\n", encoding="utf-8")
    private_dir = data_root / task_id / "prepared" / "private"
    private_dir.mkdir(parents=True)
    (private_dir / "answer.csv").write_text("output\n0\n", encoding="utf-8")
    return requested_root, data_root, public_dir


def test_policy_parser_accepts_a_subset() -> None:
    args = runner._build_parser().parse_args(
        [
            "--data-root",
            "/unused",
            "--model",
            "unit-test-model",
            "--policies",
            "main,l3",
        ]
    )

    assert args.policies == ("main", "l3")


@pytest.mark.parametrize(
    ("source_id", "expected_minimum"),
    [("dabench", 250), ("moscibench", 80)],
)
def test_bundled_descriptions_have_one_replaceable_dataset_section(
    source_id: str,
    expected_minimum: int,
) -> None:
    _, descriptor, _ = _benchmark_profile(source_id)
    descriptions = sorted(descriptor.registry_root.glob("*/description.md"))
    assert len(descriptions) >= expected_minimum
    assert descriptions

    for path in descriptions:
        replaced = replace_dataset_description(
            path.read_text(encoding="utf-8"),
            "### Level 1 Semantic Data Map\n\n- validation sentinel",
        )
        assert replaced.count("validation sentinel") == 1


@pytest.mark.parametrize(
    ("source_id", "workflow", "requires_output", "feedback_retries"),
    [
        ("dabench", "aide", False, 0),
        ("moscibench", "react", True, 2),
    ],
)
def test_catalog_capability_and_reviewed_profile_are_resolved_together(
    source_id: str,
    workflow: str,
    requires_output: bool,
    feedback_retries: int,
) -> None:
    catalog, descriptor, profile = _benchmark_profile(source_id)

    assert catalog.get_source(source_id) == descriptor
    assert descriptor.source_id == source_id
    assert descriptor.contract_id == "mle_task_contract/v1"
    assert descriptor.engine_id == "mle"
    assert profile.source_id == source_id
    assert profile.default_workflow == workflow
    output_contract = profile.config_overrides.get("output_contract", {})
    assert output_contract.get("require_output_before_completion", False) is requires_output
    assert output_contract.get("missing_output_feedback_retries", 0) == feedback_retries


@pytest.mark.parametrize("raw", ["l0", "main,main"])
def test_policy_parser_rejects_l0_and_duplicates(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner._policy_list(raw)


@pytest.mark.parametrize(
    ("source_id", "requires_output", "feedback_retries"),
    [("dabench", False, 0), ("moscibench", True, 2)],
)
def test_build_configs_uses_profile_output_contract_for_every_arm(
    tmp_path: Path,
    source_id: str,
    requires_output: bool,
    feedback_retries: int,
) -> None:
    _, _, profile = _benchmark_profile(source_id)
    conditions = runner._level_conditions(
        runner.SUPPORTED_POLICIES,
        l1_artifact_dir=tmp_path / "l1",
        l2_guidance_path=tmp_path / "l2.md",
    )
    configs, _ = engine.build_configs(
        conditions=conditions,
        run_id="unit-test-run",
        runtime=engine.ConditionRuntime(profile.default_workflow, "unit-test-model", 2),
        config_overrides=profile.config_overrides,
        dry_run=True,
    )

    assert tuple(configs) == runner.SUPPORTED_POLICIES
    for config in configs.values():
        assert config.output_contract.require_output_before_completion is requires_output
        assert config.output_contract.missing_output_feedback_retries == feedback_retries
        assert config.scheduler.run_id == "unit-test-run"
        assert config.run.parameters["output_artifact_suffix_seed"] == "unit-test-run"
        assert config.task_context.require_canonical_layout is True

def test_data_root_normalization_accepts_direct_and_competitions_layouts(
    tmp_path: Path,
) -> None:
    direct_requested, direct_root, _ = _make_prepared_source_task(
        tmp_path / "direct",
        task_id=DABENCH_TASK_ID,
    )
    nested_requested, nested_root, _ = _make_prepared_source_task(
        tmp_path / "nested",
        task_id=MOSCIBENCH_TASK_ID,
        nested_competitions=True,
    )

    assert engine.normalize_data_root(direct_requested, [DABENCH_TASK_ID]) == direct_root.resolve()
    assert (
        engine.normalize_data_root(nested_requested, [MOSCIBENCH_TASK_ID])
        == nested_root.resolve()
    )


@pytest.mark.parametrize(
    ("source_id", "task_id"),
    [("dabench", DABENCH_TASK_ID), ("moscibench", MOSCIBENCH_TASK_ID)],
)
def test_task_resolution_is_registry_driven_for_each_mle_source(
    tmp_path: Path,
    source_id: str,
    task_id: str,
) -> None:
    _, data_root, public_dir = _make_prepared_source_task(tmp_path, task_id=task_id)
    catalog, descriptor, _ = _benchmark_profile(source_id)
    registry = catalog.build_registry(descriptor, data_root=data_root)

    targets = engine.resolve_tasks(
        registry=registry,
        inline_tasks=[task_id],
        tasks_file=None,
        limit=None,
    )

    assert len(targets) == 1
    assert targets[0].task_id == task_id
    assert targets[0].dataset_id == (
        "mosci-cyclone" if source_id == "moscibench" else task_id
    )
    assert targets[0].public_dir == public_dir.resolve()
    assert targets[0].sample_submission_path == (
        public_dir / "sample_submission.csv"
    ).resolve()
    assert targets[0].description_sha256 == engine.sha256(
        targets[0].description_text.encode("utf-8")
    )


def test_task_resolution_rejects_incomplete_prepared_data(tmp_path: Path) -> None:
    _, data_root, _ = _make_prepared_source_task(
        tmp_path,
        task_id=DABENCH_TASK_ID,
    )
    (data_root / DABENCH_TASK_ID / "prepared" / "private" / "answer.csv").unlink()
    catalog, descriptor, _ = _benchmark_profile("dabench")
    registry = catalog.build_registry(descriptor, data_root=data_root)

    with pytest.raises(engine.PreflightError, match="not fully prepared"):
        engine.resolve_tasks(
            registry=registry,
            inline_tasks=[DABENCH_TASK_ID],
            tasks_file=None,
            limit=None,
        )


def test_l1_preflight_validates_public_coverage(tmp_path: Path) -> None:
    _, public_dir = _make_task_roots(tmp_path)
    artifact_dir = tmp_path / "l1"
    _write_l1_artifact(artifact_dir, "train.csv")
    target = _target(
        task_id=TASK_ID,
        dataset_id=TASK_ID,
        public_dir=public_dir,
        sample_submission_path=public_dir / "sample_submission.csv",
    )

    summary = engine.preflight_l1(artifact_dir, [target])

    assert summary["validated_task_count"] == 1
    assert summary["files"][TASK_ID]["path"] == str((artifact_dir / f"{TASK_ID}.json").resolve())

    _write_l1_artifact(artifact_dir, "missing.csv")
    with pytest.raises(engine.PreflightError, match="invalid L1 artifact"):
        engine.preflight_l1(artifact_dir, [target])


def test_l1_preflight_rejects_sample_submission(tmp_path: Path) -> None:
    _, public_dir = _make_task_roots(tmp_path)
    artifact_dir = tmp_path / "l1"
    _write_l1_artifact(artifact_dir, "sample_submission.csv")
    target = _target(
        task_id=TASK_ID,
        dataset_id=TASK_ID,
        public_dir=public_dir,
        sample_submission_path=public_dir / "sample_submission.csv",
    )

    with pytest.raises(engine.PreflightError, match="invalid L1 artifact"):
        engine.preflight_l1(artifact_dir, [target])


def test_l1_preflight_reuses_one_family_artifact_across_moscibench_tasks(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "l1"
    family_id = "mosci-cyclone"
    artifact_path = _write_l1_artifact(
        artifact_dir,
        "train.csv",
        artifact_id=family_id,
        dataset_id=family_id,
        task_id=None,
    )
    targets: list[engine.TaskTarget] = []
    for task_id in ("mosci-cyclone-1", "mosci-cyclone-2"):
        public_dir = tmp_path / "data" / task_id / "prepared" / "public"
        public_dir.mkdir(parents=True)
        (public_dir / "train.csv").write_text("value\n1\n", encoding="utf-8")
        sample = public_dir / "sample_submission.csv"
        sample.write_text("output\n0\n", encoding="utf-8")
        targets.append(
            _target(
                task_id=task_id,
                dataset_id=family_id,
                public_dir=public_dir,
                sample_submission_path=sample,
            )
        )

    summary = engine.preflight_l1(artifact_dir, targets)

    expected_path = str(artifact_path.resolve())
    assert summary["validated_task_count"] == 2
    assert summary["artifact_count"] == 1
    assert summary["files"]["mosci-cyclone-1"]["path"] == expected_path
    assert summary["files"]["mosci-cyclone-2"]["path"] == expected_path


def test_l2_preflight_delegates_to_core_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guidance_path = tmp_path / "guidance.md"
    guidance_path.write_text("## Guidance\n\nInspect the public data.\n", encoding="utf-8")
    calls: list[Path] = []

    def load_l2_artifact(path: Path | str) -> object:
        calls.append(Path(path).resolve())
        return object()

    monkeypatch.setattr(task_context_boundary, "load_l2_artifact", load_l2_artifact)

    resolved, summary = engine.preflight_l2(guidance_path)

    assert calls == [guidance_path.resolve()]
    assert resolved == guidance_path.resolve()
    assert summary["path"] == str(guidance_path.resolve())


def test_main_only_dry_run_needs_no_artifacts_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, data_root, _ = _make_prepared_source_task(
        tmp_path,
        task_id=DABENCH_TASK_ID,
    )
    runs_root = tmp_path / "runs"
    build_arguments: dict[str, Any] = {}

    monkeypatch.setattr(runner, "RUNS_ROOT", runs_root)
    original_build_configs = runner.build_configs

    def build_configs(**kwargs: Any) -> tuple[dict[str, Any], str]:
        build_arguments.update(kwargs)
        return original_build_configs(**kwargs)

    monkeypatch.setattr(runner, "build_configs", build_configs)

    args = runner._build_parser().parse_args(
        [
            "--data-root",
            str(data_root),
            "--tasks",
            DABENCH_TASK_ID,
            "--policies",
            "main",
            "--model",
            "unit-test-model",
            "--dry-run",
        ]
    )

    assert runner._execute(args) is None
    conditions = build_arguments["conditions"]
    assert [condition.condition_id for condition in conditions] == ["main"]
    assert conditions[0].l1_artifact_dir is None
    assert conditions[0].l2_guidance_path is None
    assert not runs_root.exists()

    plan = json.loads(capsys.readouterr().out)
    assert plan["policies"] == ["main"]
    assert plan["artifacts"] == {}
    assert plan["context_audit"]["status"] == "not_run"
    assert plan["benchmark"]["source_id"] == "dabench"
    assert plan["selection"]["tasks"] == [DABENCH_TASK_ID]


def test_runtime_provenance_is_collected_and_fixed_context_is_verified() -> None:
    base = {
        "policy": "main",
        "dataset_id": TASK_ID,
        "main_data_report_sha256": "report-hash",
        "l1": None,
        "l2": None,
    }
    fixed = {
        "io_instructions_sha256": "io-hash",
        "output_artifact_name": "submission.csv",
        "submission_contract_sha256": "submission-hash",
        "evaluation_contract_ref_sha256": "evaluation-hash",
    }
    benchmark = SimpleNamespace(
        runner=SimpleNamespace(
            get_run_records=lambda: [
                {
                    "task_id": TASK_ID,
                    "summary": {"success": True},
                    "task_context_audit": {
                        "provenance": base,
                        "fixed_context": fixed,
                    },
                }
            ]
        )
    )

    collected = engine.collect_task_context_audit(
        benchmark,
        tasks=[TASK_ID],
        policy="main",
    )
    reference: dict[str, dict[str, str]] = {}
    engine.verify_fixed_context(collected, reference, condition_id="main")

    treatment = {
        TASK_ID: {
            "task_context_provenance": {
                **base,
                "policy": "l1",
                "l1": {"path": "/l1.json", "sha256": "l1-hash"},
            },
            "fixed_task_context": fixed,
        }
    }
    engine.verify_fixed_context(treatment, reference, condition_id="l1")

    changed = {
        TASK_ID: {
            "task_context_provenance": {**base, "policy": "l2"},
            "fixed_task_context": {
                **fixed,
                "io_instructions_sha256": "changed-io",
            },
        }
    }
    with pytest.raises(engine.ExperimentInvariantError, match="io_instructions_sha256"):
        engine.verify_fixed_context(changed, reference, condition_id="l2")


def test_runtime_provenance_is_required_for_every_selected_task() -> None:
    benchmark = SimpleNamespace(
        runner=SimpleNamespace(
            get_run_records=lambda: [{"task_id": TASK_ID, "summary": {"success": True}}]
        )
    )

    with pytest.raises(engine.ExperimentInvariantError, match="did not persist"):
        engine.collect_task_context_audit(
            benchmark,
            tasks=[TASK_ID],
            policy="main",
        )


def test_runtime_provenance_requires_dataset_identity() -> None:
    benchmark = SimpleNamespace(
        runner=SimpleNamespace(
            get_run_records=lambda: [
                {
                    "task_id": TASK_ID,
                    "summary": {"success": True},
                    "task_context_audit": {
                        "provenance": {
                            "policy": "main",
                            "main_data_report_sha256": "report-hash",
                            "l1": None,
                            "l2": None,
                        },
                        "fixed_context": {
                            "io_instructions_sha256": "io-hash",
                            "output_artifact_name": "submission.csv",
                        },
                    },
                }
            ]
        )
    )

    with pytest.raises(engine.ExperimentInvariantError, match="dataset identity"):
        engine.collect_task_context_audit(
            benchmark,
            tasks=[TASK_ID],
            policy="main",
        )


def test_identical_retry_records_are_deduplicated_but_conflicts_fail() -> None:
    provenance = {
        "policy": "main",
        "dataset_id": TASK_ID,
        "main_data_report_sha256": "report-hash",
        "l1": None,
        "l2": None,
    }
    record = {
        "task_id": TASK_ID,
        "summary": {"success": True},
        "task_context_audit": {
            "provenance": provenance,
            "fixed_context": {
                "io_instructions_sha256": "io-hash",
                "output_artifact_name": "submission.csv",
            },
        },
    }
    records = [
        {
            **record,
            "summary": {"success": False, "result": "[ERROR] retryable failure"},
        },
        dict(record),
    ]
    benchmark = SimpleNamespace(runner=SimpleNamespace(get_run_records=lambda: records))

    collected = engine.collect_task_context_audit(
        benchmark,
        tasks=[TASK_ID],
        policy="main",
    )
    assert list(collected) == [TASK_ID]
    assert collected[TASK_ID]["run_summary"]["success"] is True
    assert collected[TASK_ID]["run_summary"]["attempt_count"] == 2

    records[1] = {
        **record,
        "task_context_audit": {
            **record["task_context_audit"],
            "fixed_context": {
                **record["task_context_audit"]["fixed_context"],
                "output_artifact_name": "changed.csv",
            },
        },
    }
    with pytest.raises(engine.ExperimentInvariantError, match="conflicting retry"):
        engine.collect_task_context_audit(
            benchmark,
            tasks=[TASK_ID],
            policy="main",
        )


def test_failed_task_record_is_audited_as_an_outcome() -> None:
    benchmark = SimpleNamespace(
        runner=SimpleNamespace(
            get_run_records=lambda: [
                {
                    "task_id": TASK_ID,
                    "summary": {
                        "success": False,
                        "result": "[ERROR] workflow failed after spec construction",
                    },
                    "task_context_audit": {
                        "provenance": {
                            "policy": "main",
                            "dataset_id": TASK_ID,
                            "main_data_report_sha256": "report-hash",
                            "l1": None,
                            "l2": None,
                        },
                        "fixed_context": {
                            "io_instructions_sha256": "io-hash",
                            "output_artifact_name": "submission.csv",
                        },
                    },
                }
            ]
        )
    )

    collected = engine.collect_task_context_audit(
        benchmark,
        tasks=[TASK_ID],
        policy="main",
    )

    assert collected[TASK_ID]["run_summary"] == {
        "success": False,
        "result": "[ERROR] workflow failed after spec construction",
        "attempt_count": 1,
    }


@pytest.mark.parametrize(
    ("workflow_success", "valid_submission", "expected_status"),
    [
        (True, True, "completed"),
        (True, False, "invalid_submission"),
        (False, False, "workflow_failed"),
    ],
)
def test_task_outcomes_distinguish_workflow_and_submission_failures(
    workflow_success: bool,
    valid_submission: bool,
    expected_status: str,
) -> None:
    columns = [
        "competition_id",
        "score",
        "submission_exists",
        "valid_submission",
        "error_message",
    ]
    benchmark = SimpleNamespace(
        get_result_columns=lambda: columns,
        results=[
            (
                TASK_ID,
                0.5 if valid_submission else None,
                valid_submission,
                valid_submission,
                None if valid_submission else "missing or invalid submission",
            )
        ],
    )
    audit = {
        TASK_ID: {
            "run_summary": {
                "success": workflow_success,
                "result": None,
                "attempt_count": 1,
            }
        }
    }

    outcomes = engine.collect_task_outcomes(
        benchmark,
        audit,
        tasks=[TASK_ID],
        condition_id="main",
    )

    assert outcomes[TASK_ID]["status"] == expected_status


@pytest.mark.parametrize(
    (
        "source_id",
        "task_id",
        "expected_workflow",
        "requires_output",
        "valid_submission",
        "expected_status",
    ),
    [
        (
            "dabench",
            DABENCH_TASK_ID,
            "aide",
            False,
            True,
            "completed",
        ),
        (
            "moscibench",
            MOSCIBENCH_TASK_ID,
            "react",
            True,
            False,
            "completed_with_task_failures",
        ),
    ],
)
def test_execute_uses_dynamic_benchmark_contract_and_persists_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_id: str,
    task_id: str,
    expected_workflow: str,
    requires_output: bool,
    valid_submission: bool,
    expected_status: str,
) -> None:
    requested_root, data_root, public_dir = _make_prepared_source_task(
        tmp_path,
        task_id=task_id,
        nested_competitions=True,
    )
    runs_root = tmp_path / "runs"
    _, descriptor, _ = _benchmark_profile(source_id)
    provenance = {
        "policy": "main",
        "dataset_id": ("mosci-cyclone" if source_id == "moscibench" else task_id),
        "main_data_report_sha256": "report-hash",
        "l1": None,
        "l2": None,
    }
    fixed_context = {
        "io_instructions_sha256": "io-hash",
        "output_artifact_name": "submission.csv",
        "submission_contract_sha256": "submission-hash",
        "evaluation_contract_ref_sha256": "evaluation-hash",
    }
    columns = [
        "competition_id",
        "score",
        "submission_exists",
        "valid_submission",
        "error_message",
    ]
    result = SimpleNamespace(
        runner=SimpleNamespace(
            get_run_records=lambda: [
                {
                    "task_id": task_id,
                    "summary": {"success": True, "result": "finished"},
                    "task_context_audit": {
                        "provenance": provenance,
                        "fixed_context": fixed_context,
                    },
                }
            ]
        ),
        get_result_columns=lambda: columns,
        results=[
            (
                task_id,
                0.5 if valid_submission else None,
                valid_submission,
                valid_submission,
                None if valid_submission else "missing submission",
            )
        ],
        results_path=tmp_path / "results.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    constructor_calls: list[dict[str, object]] = []
    run_calls: list[dict[str, object]] = []

    class FakeDSBenchmark:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)

        def run(self, **kwargs: object) -> object:
            run_calls.append(kwargs)
            return result

    monkeypatch.setattr(runner, "RUNS_ROOT", runs_root)
    original_build_configs = runner.build_configs

    def build_configs_without_api_key(**kwargs: Any) -> tuple[dict[str, Any], str]:
        kwargs["dry_run"] = True
        return original_build_configs(**kwargs)

    monkeypatch.setattr(runner, "build_configs", build_configs_without_api_key)
    monkeypatch.setattr("dslighting.api.benchmark.DSBenchmark", FakeDSBenchmark)
    args = runner._build_parser().parse_args(
        [
            "--benchmark",
            source_id,
            "--data-root",
            str(requested_root),
            "--tasks",
            task_id,
            "--policies",
            "main",
            "--model",
            "unit-test-model",
        ]
    )

    manifest_path = runner._execute(args)

    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == expected_status
    assert manifest["runs"][0]["status"] == expected_status
    assert manifest["benchmark"]["source_id"] == source_id
    assert manifest["benchmark"]["contract_id"] == "mle_task_contract/v1"
    assert manifest["benchmark"]["engine_id"] == "mle"
    assert manifest["benchmark"]["data_root"] == str(data_root.resolve())
    assert manifest["benchmark"]["registry_root"] == str(descriptor.registry_root)
    assert manifest["benchmark"]["source_manifest"]["path"] == str(descriptor.manifest_path)
    assert manifest["selection"]["tasks"] == [task_id]
    assert manifest["selection"]["public_dirs"] == {task_id: str(public_dir.resolve())}
    assert manifest["runtime"]["workflow"] == expected_workflow
    assert (
        manifest["runtime"]["output_contract"]["require_output_before_completion"]
        is requires_output
    )
    assert (
        manifest["runs"][0]["task_context_audit"][task_id]["task_context_provenance"] == provenance
    )
    assert manifest["runs"][0]["task_outcomes"][task_id]["valid_submission"] is valid_submission
    assert constructor_calls == [
        {
            "benchmark_type": source_id,
            "exp_name": f"{manifest['run_id']}_main",
            "data_dir": str(data_root.resolve()),
            "vendor_comp_dir": str(descriptor.registry_root),
            "competitions": [task_id],
        }
    ]
    assert len(run_calls) == 1
    assert run_calls[0]["log_path"].endswith("/main")
    assert run_calls[0]["verbose"] is True
    assert "task_data_views" not in run_calls[0]["config"].run.parameters


def test_runtime_artifacts_must_match_preflight_digests() -> None:
    artifact = {"path": "/l1.json", "sha256": "l1-hash"}
    context = {
        TASK_ID: {
            "task_context_provenance": {
                "policy": "l1",
                "dataset_id": TASK_ID,
                "main_data_report_sha256": "report-hash",
                "l1": artifact,
                "l2": None,
            },
            "fixed_task_context": {
                "io_instructions_sha256": "io-hash",
                "output_artifact_name": "submission.csv",
            },
        }
    }

    engine.verify_selected_artifacts(
        context,
        condition=engine.ExperimentCondition("l1", "l1"),
        l1_summary={"files": {TASK_ID: {"artifact_id": TASK_ID, **artifact}}},
        l2_summary=None,
    )

    with pytest.raises(engine.ExperimentInvariantError, match="preflighted L1"):
        engine.verify_selected_artifacts(
            context,
            condition=engine.ExperimentCondition("l1", "l1"),
            l1_summary={"files": {TASK_ID: {"path": "/l1.json", "sha256": "changed-hash"}}},
            l2_summary=None,
        )
