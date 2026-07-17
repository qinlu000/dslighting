from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.data_card_ablation import prepare_dabench_perception_data as dabench_clean
from experiments.data_card_ablation.engine import (
    ConditionRuntime,
    ExperimentCondition,
    PreflightError,
    TaskTarget,
    build_configs,
    preflight_l1,
)
from experiments.data_card_ablation.perception_skills import dabench_adapter
from experiments.data_card_ablation.perception_skills.profile import (
    PerceptionSkillCondition,
    load_profile,
)
from experiments.data_card_ablation.perception_skills.runner import load_perception_skills

TEST_RAW_PAYLOAD = b"feature,value\na,1\nb,2\n"
TEST_RAW_SHA256 = hashlib.sha256(TEST_RAW_PAYLOAD).hexdigest()
TEST_FAMILY_ID = f"dabench-family-{TEST_RAW_SHA256}"


def _write_skills_manifest(
    root: Path,
    conditions: list[dict[str, object]],
) -> tuple[Path, Path]:
    skills = root / "skills"
    annotations = root / "annotations"
    skills.mkdir()
    annotations.mkdir()
    manifest = skills / "perception_skills.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "perception_skills_v1",
                "perception_skills": conditions,
            }
        ),
        encoding="utf-8",
    )
    return manifest, annotations


def test_repository_manifest_resolves_the_frozen_mosci_conditions() -> None:
    profile = load_profile("moscibench")
    conditions, summary = load_perception_skills(
        profile.skills_manifest,
        profile.annotations_root,
        tuple(profile.conditions[1:]),
    )

    assert [condition.perception_skill_id for condition in conditions] == [
        "no-added-skill-v1",
        "dataset-semantics-v1",
        "scientific-modalities-v1",
    ]
    assert [condition.is_control for condition in conditions] == [True, False, False]
    assert summary["schema_version"] == "perception_skills_v1"


def test_skill_manifest_rejects_duplicate_keys_and_entrypoint_escape(tmp_path: Path) -> None:
    manifest, annotations = _write_skills_manifest(tmp_path, [])
    manifest.write_text(
        '{"schema_version":"perception_skills_v1",'
        '"schema_version":"perception_skills_v1","perception_skills":[]}',
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="duplicate JSON key"):
        load_perception_skills(manifest, annotations, ("semantic",))

    (tmp_path / "outside.md").write_text("skill", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "perception_skills_v1",
                "perception_skills": [
                    {
                        "perception_skill_id": "semantic",
                        "skill_entrypoints": ["../outside.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (annotations / "semantic").mkdir()
    with pytest.raises(PreflightError, match="escapes"):
        load_perception_skills(manifest, annotations, ("semantic",))


def test_shared_config_builder_changes_only_declared_treatment(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    runtime = ConditionRuntime(
        workflow="react",
        model="openai/test-model",
        task_concurrency=4,
        llm_global_concurrency=20,
        llm_per_key_concurrency=20,
        llm_max_retries=10,
        llm_thinking=False,
        sandbox_timeout_seconds=7200,
        sandbox_backend="bubblewrap",
        sandbox_environment_policy="allowlist",
        sandbox_network_policy="disabled",
        device_admission="cpu_default",
        checkpoint_resume_enabled=True,
    )
    conditions = (
        ExperimentCondition("main", "main"),
        ExperimentCondition("semantic", "l1", l1_artifact_dir=annotations),
    )

    configs = build_configs(
        conditions=conditions,
        run_id="test-run",
        runtime=runtime,
        config_overrides=None,
        dry_run=True,
    )
    repeated_configs = build_configs(
        conditions=conditions,
        run_id="test-run-r02",
        runtime=runtime,
        config_overrides=None,
        dry_run=True,
    )

    assert repeated_configs["main"].scheduler.run_id == "test-run-r02"
    assert repeated_configs["main"].run.parameters["output_artifact_suffix_seed"] == "test-run-r02"
    assert configs["main"].task_context.policy == "main"
    assert configs["semantic"].task_context.policy == "l1"
    assert configs["semantic"].task_context.l1_artifact_dir == str(annotations)
    assert configs["main"].output_contract == configs["semantic"].output_contract
    assert configs["main"].scheduler.llm_max_concurrency == 20
    assert configs["main"].llm.max_retries == 10
    assert configs["main"].llm.thinking is False


def _make_family_target(root: Path, task_id: str = "dabench-family-task") -> TaskTarget:
    task_root = root / "clean" / task_id
    public = task_root / "prepared" / "public"
    private = task_root / "prepared" / "private"
    raw = task_root / "raw"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    raw.mkdir(parents=True)
    (public / "train.csv").write_bytes(TEST_RAW_PAYLOAD)
    sample = public / "sample_submission.csv"
    sample.write_text("answer\n0\n", encoding="utf-8")
    (private / "answer.csv").write_text("answer\n1\n", encoding="utf-8")
    (raw / "source.csv").write_bytes(TEST_RAW_PAYLOAD)
    return TaskTarget(
        task_id=task_id,
        dataset_id=task_id,
        public_dir=public,
        sample_submission_path=sample,
        description_text=(
            "# Analysis\n\n## Data Description\n\nOriginal description.\n\n"
            "## Task\n\nCalculate the requested result.\n"
        ),
        description_sha256="0" * 64,
    )


def _write_family_manifest(root: Path, target: TaskTarget) -> Path:
    public_sha = hashlib.sha256((target.public_dir / "train.csv").read_bytes()).hexdigest()
    sample_sha = hashlib.sha256(target.sample_submission_path.read_bytes()).hexdigest()  # type: ignore[union-attr]
    private = target.public_dir.parent / "private" / "answer.csv"
    private_sha = hashlib.sha256(private.read_bytes()).hexdigest()
    vendor_contract = json.dumps(
        {"schema_version": dabench_clean.VENDOR_CONTRACT_SCHEMA_VERSION, "files": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    clean_root = target.public_dir.parents[2]
    path = clean_root / dabench_clean.MANIFEST_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": dabench_adapter.DATASET_FAMILY_MANIFEST_SCHEMA_VERSION,
                "source_root": "/source",
                "clean_root": str(clean_root.resolve()),
                "task_count": 1,
                "family_count": 1,
                "tasks": [
                    {
                        "task_id": target.task_id,
                        "family_id": TEST_FAMILY_ID,
                        "raw_filename": "source.csv",
                        "raw_sha256": TEST_RAW_SHA256,
                        "public_train_sha256": public_sha,
                        "sample_submission_sha256": sample_sha,
                        "private_answer_sha256": private_sha,
                        "vendor_contract_sha256": hashlib.sha256(vendor_contract).hexdigest(),
                        "vendor_contract_files": [],
                    }
                ],
                "families": [
                    {
                        "family_id": TEST_FAMILY_ID,
                        "raw_sha256": TEST_RAW_SHA256,
                        "public_train_sha256": public_sha,
                        "representative_task_id": target.task_id,
                        "task_ids": [target.task_id],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_family_annotation(root: Path) -> Path:
    root.mkdir(parents=True)
    path = root / f"{TEST_FAMILY_ID}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "l1_semantic_map_v1",
                "dataset_id": TEST_FAMILY_ID,
                "data_objects": [
                    {
                        "name": "measurements",
                        "files": ["train.csv"],
                        "kind": "table",
                        "meaning": "Named features and their recorded values.",
                    }
                ],
                "variables": [
                    {
                        "object": "measurements",
                        "file": "train.csv",
                        "name": "feature",
                        "meaning": "Name of the measured feature.",
                        "unit": None,
                    }
                ],
                "structure": [],
                "uncertainties": [],
                "annotation_notes": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dabench_family_manifest_resolves_selected_tasks(tmp_path: Path) -> None:
    target = _make_family_target(tmp_path)
    manifest = _write_family_manifest(tmp_path, target)

    mapping, summary = dabench_adapter._load_dataset_family_manifest(manifest, [target])

    assert mapping == {target.task_id: TEST_FAMILY_ID}
    assert summary["task_to_family"] == mapping
    assert summary["selected_task_count"] == 1


def test_family_annotation_materialization_preserves_solver_visible_content(
    tmp_path: Path,
) -> None:
    target = _make_family_target(tmp_path)
    manifest = _write_family_manifest(tmp_path, target)
    mapping, _ = dabench_adapter._load_dataset_family_manifest(manifest, [target])
    canonical_dir = tmp_path / "annotations" / "no-added-skill-v1"
    _write_family_annotation(canonical_dir)
    skill = PerceptionSkillCondition("no-added-skill-v1", (), canonical_dir)
    dabench_adapter._preflight_family_l1(
        canonical_dir,
        [target],
        mapping,
    )

    runtime_skills, resolved = dabench_adapter._materialize_family_annotations(
        (skill,),
        targets=[target],
        task_to_family=mapping,
        run_root=tmp_path / "run",
    )

    resolved_dir = runtime_skills[0].annotations_dir
    runtime_summary = preflight_l1(resolved_dir, [target])
    item = resolved["conditions"][skill.perception_skill_id]["files"][target.task_id]
    assert item["family_id"] == TEST_FAMILY_ID
    assert runtime_summary["files"][target.task_id]["artifact_id"] == target.task_id
