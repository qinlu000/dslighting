from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.data_card_ablation import run_perception_skills as runner


def _write_manifest(
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


def test_parser_supports_defaults_named_selection_and_excluding_main() -> None:
    defaults = runner._build_parser().parse_args(
        ["--data-root", "/tmp/data", "--dry-run"]
    )

    assert defaults.benchmark == "moscibench"
    assert defaults.perception_skills is None
    assert defaults.exclude_main is False
    assert defaults.model is None
    assert defaults.perception_skills_manifest == runner.DEFAULT_SKILLS_MANIFEST
    assert defaults.annotations_root == runner.DEFAULT_ANNOTATIONS_ROOT

    named = runner._build_parser().parse_args(
        [
            "--data-root",
            "/tmp/data",
            "--perception-skills",
            "no-added-skill-v1,dataset-semantics-v1",
        ]
    )
    assert named.perception_skills == (
        "no-added-skill-v1",
        "dataset-semantics-v1",
    )

    without_main = runner._build_parser().parse_args(
        ["--data-root", "/tmp/data", "--exclude-main"]
    )
    assert without_main.exclude_main is True


@pytest.mark.parametrize(
    "raw",
    ["", "same,same", "../escape", "nested/skill"],
)
def test_perception_skill_list_rejects_invalid_selection(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner._perception_skill_list(raw)


def test_repository_manifest_resolves_three_conditions() -> None:
    conditions, summary = runner._load_perception_skills(
        runner.DEFAULT_SKILLS_MANIFEST,
        runner.DEFAULT_ANNOTATIONS_ROOT,
        None,
    )

    assert [condition.perception_skill_id for condition in conditions] == [
        "no-added-skill-v1",
        "dataset-semantics-v1",
        "scientific-modalities-v1",
    ]
    assert [condition.is_control for condition in conditions] == [True, False, False]
    assert summary["schema_version"] == "perception_skills_v1"
    assert all(condition.annotations_dir.is_dir() for condition in conditions)
    assert len(summary["selected"][1]["skill_entrypoints"]) == 1
    assert len(summary["selected"][2]["skill_entrypoints"]) == 2


def test_manifest_selection_preserves_requested_order() -> None:
    conditions, _ = runner._load_perception_skills(
        runner.DEFAULT_SKILLS_MANIFEST,
        runner.DEFAULT_ANNOTATIONS_ROOT,
        ("scientific-modalities-v1", "no-added-skill-v1"),
    )

    assert [condition.perception_skill_id for condition in conditions] == [
        "scientific-modalities-v1",
        "no-added-skill-v1",
    ]


def test_manifest_rejects_unknown_selection() -> None:
    with pytest.raises(runner.PreflightError, match="unknown perception skill"):
        runner._load_perception_skills(
            runner.DEFAULT_SKILLS_MANIFEST,
            runner.DEFAULT_ANNOTATIONS_ROOT,
            ("not-registered",),
        )


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    annotations = tmp_path / "annotations"
    skills.mkdir()
    annotations.mkdir()
    manifest = skills / "perception_skills.json"
    manifest.write_text(
        '{"schema_version":"perception_skills_v1",'
        '"schema_version":"perception_skills_v1","perception_skills":[]}',
        encoding="utf-8",
    )

    with pytest.raises(runner.PreflightError, match="duplicate JSON key"):
        runner._load_perception_skills(manifest, annotations, None)


def test_manifest_rejects_entrypoint_escape(tmp_path: Path) -> None:
    manifest, annotations = _write_manifest(
        tmp_path,
        [
            {
                "perception_skill_id": "unsafe-skill",
                "skill_entrypoints": ["../outside.md"],
            }
        ],
    )
    (annotations / "unsafe-skill").mkdir()

    with pytest.raises(runner.PreflightError, match="escapes its declared root"):
        runner._load_perception_skills(manifest, annotations, None)


def test_manifest_requires_annotation_directory(tmp_path: Path) -> None:
    manifest, annotations = _write_manifest(
        tmp_path,
        [
            {
                "perception_skill_id": "missing-annotations",
                "skill_entrypoints": [],
            }
        ],
    )

    with pytest.raises(runner.PreflightError, match="annotations is not a readable directory"):
        runner._load_perception_skills(manifest, annotations, None)


def test_build_configs_adds_main_and_changes_only_treatment_fields(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    perception_skills = (
        runner.PerceptionSkillCondition("first", (), first_dir),
        runner.PerceptionSkillCondition("second", (tmp_path / "SKILL.md",), second_dir),
    )
    conditions = runner._build_downstream_conditions(
        perception_skills,
        include_main=True,
    )

    configs, fixed_sha256 = runner._build_configs(
        conditions=conditions,
        run_id="test-run",
        workflow="react",
        model="openai/test-model",
        concurrency=3,
        config_overrides=None,
        dry_run=True,
    )

    assert list(configs) == ["main", "first", "second"]
    assert len(fixed_sha256) == 64
    assert configs["main"].task_context.policy == "main"
    assert configs["main"].task_context.l1_artifact_dir is None
    assert configs["first"].task_context.policy == "l1"
    assert configs["second"].task_context.policy == "l1"
    assert configs["first"].task_context.l1_artifact_dir == str(first_dir)
    assert configs["second"].task_context.l1_artifact_dir == str(second_dir)
    assert all(config.task_context.l2_guidance_path is None for config in configs.values())
    assert all(config.scheduler.max_concurrency == 3 for config in configs.values())
    assert all(config.scheduler.llm_max_concurrency == 3 for config in configs.values())
    assert all(config.scheduler.gpu_policy == "cpu_default" for config in configs.values())
    assert all(config.llm.max_concurrent_per_key == 3 for config in configs.values())
    assert all(
        config.run.parameters["output_artifact_suffix_seed"] == "test-run"
        for config in configs.values()
    )


def test_fixed_config_fingerprint_detects_downstream_change(tmp_path: Path) -> None:
    perception_skills = (
        runner.PerceptionSkillCondition("first", (), tmp_path),
        runner.PerceptionSkillCondition("second", (), tmp_path),
    )
    conditions = runner._build_downstream_conditions(
        perception_skills,
        include_main=True,
    )
    configs, _ = runner._build_configs(
        conditions=conditions,
        run_id="test-run",
        workflow="react",
        model="openai/test-model",
        concurrency=1,
        config_overrides=None,
        dry_run=True,
    )

    configs["second"].scheduler.max_concurrency = 2

    assert runner._fixed_config_fingerprint(configs["first"]) != runner._fixed_config_fingerprint(
        configs["second"]
    )


def test_downstream_conditions_can_exclude_main() -> None:
    skill = runner.PerceptionSkillCondition(
        "no-added-skill-v1",
        (),
        runner.DEFAULT_ANNOTATIONS_ROOT / "no-added-skill-v1",
    )

    conditions = runner._build_downstream_conditions((skill,), include_main=False)

    assert [condition.condition_id for condition in conditions] == ["no-added-skill-v1"]
    assert conditions[0].task_context_policy == "l1"
