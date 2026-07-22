from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.data_card_ablation.engine import ConditionExperimentEngine, ManifestStore
from experiments.data_card_ablation.perception_skills.profile import (
    ExperimentRequest,
    ExperimentSelection,
    PerceptionSkillCondition,
    ProtocolOverrides,
    load_profile,
)
from experiments.data_card_ablation.perception_skills.runner import (
    ExperimentConfigurationError,
    PerceptionSkillsRunner,
    _conditions,
    _runtime,
)
from experiments.data_card_ablation.perception_skills.runtime import (
    PROXY_VARIABLES,
    DataContainedRuntime,
)


def test_profiles_freeze_benchmark_specific_conditions_and_runtime() -> None:
    dabench = load_profile("dabench")
    mosci = load_profile("perception-skills-moscibench-v2")

    assert dabench.conditions == ("main", "no-added-skill-v1", "dataset-semantics-v1")
    assert dabench.annotations_root.is_relative_to(
        Path(__file__).resolve().parents[2] / "experiments" / "data_card_ablation" / "artifacts"
    )
    assert not hasattr(dabench, "annotation_input_root")
    assert not hasattr(dabench, "frozen_input_sha256")
    assert mosci.conditions[-1] == "scientific-modalities-v1"
    assert dabench.profile_id == "perception-skills-dabench-v3"
    assert dabench.runtime.task_concurrency == 257
    assert dabench.runtime.llm_global_concurrency == 257
    assert dabench.runtime.llm_per_key_concurrency == 257
    assert mosci.runtime.task_concurrency == 20
    assert mosci.runtime.llm_global_concurrency == 20
    assert mosci.runtime.llm_per_key_concurrency == 20
    for profile in (dabench, mosci):
        assert profile.runtime.workflow == "react"
        assert profile.runtime.model == "openai/DeepSeek-V4-Flash"
        assert profile.runtime.llm_max_retries == 10
        assert profile.runtime.llm_thinking is False
        assert profile.runtime.sandbox_timeout_seconds == 7200
        assert profile.runtime.sandbox_backend == "bubblewrap"
        assert profile.runtime.sandbox_network_policy == "disabled"
        assert len(profile.sha256) == 64


def test_perception_conditions_are_only_declarations_for_the_shared_engine(
    tmp_path: Path,
) -> None:
    control = PerceptionSkillCondition("no-skill", (), tmp_path / "control")
    semantic = PerceptionSkillCondition(
        "semantic",
        (tmp_path / "SKILL.md",),
        tmp_path / "semantic",
    )

    conditions = _conditions((control, semantic))

    assert [condition.condition_id for condition in conditions] == [
        "main",
        "no-skill",
        "semantic",
    ]
    assert [condition.task_context_policy for condition in conditions] == [
        "main",
        "l1",
        "l1",
    ]
    assert conditions[1].control is True
    assert conditions[2].control is False
    assert ConditionExperimentEngine.__module__.endswith("data_card_ablation.engine")


def test_runtime_profile_maps_once_to_the_shared_engine() -> None:
    runtime = _runtime(load_profile("dabench"))

    assert runtime.task_concurrency == 257
    assert runtime.llm_global_concurrency == 257
    assert runtime.sandbox_backend == "bubblewrap"
    assert runtime.checkpoint_resume_enabled is True


def test_old_double_runner_modules_are_removed() -> None:
    package = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "data_card_ablation"
        / "perception_skills"
    )

    for name in ("experiment.py", "single_run.py", "manifest_store.py", "models.py"):
        assert not (package / name).exists()
    runner_source = (package / "runner.py").read_text(encoding="utf-8")
    assert "SingleRunRequest" not in runner_source
    assert "subprocess.Popen" not in runner_source
    assert "ConditionExperimentEngine" in runner_source


def test_runtime_confines_state_to_data_and_removes_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in PROXY_VARIABLES:
        monkeypatch.setenv(name, "http://proxy.invalid")

    environment = DataContainedRuntime(project).environment_for("run-1")

    for name in PROXY_VARIABLES:
        assert name not in environment
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "UV_TOOL_DIR"):
        assert Path(environment[name]).is_relative_to(project / "data")
    assert environment["UV_OFFLINE"] == "1"
    assert environment["PIP_NO_INDEX"] == "1"


def test_overrides_require_explicit_noncanonical_gate() -> None:
    runner = PerceptionSkillsRunner(load_profile("dabench"))
    request = ExperimentRequest(
        execute=False,
        repetitions=1,
        selection=ExperimentSelection(),
        overrides=ProtocolOverrides(model="different-model"),
    )

    with pytest.raises(ExperimentConfigurationError, match="allow-protocol-override"):
        runner._validate_request(request)


def test_resume_targets_one_batch_manifest(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "batch")
    store.write({"schema_version": "perception_skills_experiment_v2", "runs": []})

    assert store.read()["runs"] == []
    assert store.path == (tmp_path / "batch" / "manifest.json").resolve()


def test_resume_checks_only_the_explicit_experiment_identity(tmp_path: Path) -> None:
    profile = load_profile("dabench")
    conditions = _conditions(
        (PerceptionSkillCondition("no-added-skill-v1", (), tmp_path / "annotations"),)
    )
    manifest = {
        "benchmark": {"source_id": "dabench"},
        "selection": {"tasks": ["task-1"]},
        "conditions": [condition.as_manifest() for condition in conditions],
        "repetitions": 5,
    }

    PerceptionSkillsRunner._validate_resume_manifest(
        manifest,
        profile=profile,
        prepared=SimpleNamespace(tasks=("task-1",)),
        conditions=conditions,
        repetitions=5,
    )

    manifest["selection"] = {"tasks": ["task-2"]}
    with pytest.raises(ExperimentConfigurationError, match="does not match"):
        PerceptionSkillsRunner._validate_resume_manifest(
            manifest,
            profile=profile,
            prepared=SimpleNamespace(tasks=("task-1",)),
            conditions=conditions,
            repetitions=5,
        )


def test_public_launcher_works_outside_the_repository(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [str(project_root / "perception-skills"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "perception-skills" in result.stdout
