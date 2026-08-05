from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dslighting.config import AgentRuntimeConfig
from dslighting.core.config.builder import ConfigBuilder
from dslighting.error import ConfigurationError
from dslighting.prompts.workflows.react import create_react_prompt
from dslighting.workflows.factory.builtin import _resolve_agent_skill
from dslighting.workflows.search.react.workflow import ReActWorkflow
from experiments.datacope_dabench.adapter import (
    export_workspace_run,
    prepare_public_data,
    submission_signature,
)
from experiments.datacope_dabench.datacope_bridge import (
    group_predictions_exactly,
    skill_output_contract,
    validate_prediction_runs,
    validate_public_data,
)
from experiments.datacope_dabench.protocol import (
    PAPER_DISCOVERY_ROUNDS,
    PAPER_EVALUATION_TEMPERATURE,
    PAPER_EXPLORE_TEMPERATURE,
    PAPER_REACT_MAX_STEPS,
    PAPER_TRAJECTORIES_PER_TASK,
)
from experiments.datacope_dabench.runner import resolve_run_spec
from experiments.datacope_dabench.sidecar import (
    CodexRuntime,
    _codex_runtime,
    _discovery_command,
    _discovery_environment,
    run_sidecar,
)
from experiments.datacope_dabench.split import (
    DEFAULT_FAMILY_MANIFEST,
    SOURCE_MANIFEST_SHA256,
    load_dabench_split,
)


def test_submission_signature_normalizes_csv_encoding_and_line_endings(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b'id,answer\r\n1,"value"\r\n')
    second.write_text("\ufeffid,answer\n1,value\n", encoding="utf-8")

    first_signature = submission_signature(first)
    second_signature = submission_signature(second)

    assert first_signature["status"] == "valid"
    assert first_signature["prediction"] == second_signature["prediction"]
    assert first_signature["rows"] == [["id", "answer"], ["1", "value"]]


def test_frozen_split_is_family_disjoint_and_covers_all_tasks() -> None:
    split = load_dabench_split()

    assert split.source_manifest_sha256 == SOURCE_MANIFEST_SHA256
    assert len(split.explore_family_ids) == 13
    assert len(split.test_family_ids) == 38
    assert set(split.explore_family_ids).isdisjoint(split.test_family_ids)
    assert len(split.explore_task_ids) == 64
    assert len(split.test_task_ids) == 193
    assert set(split.explore_task_ids).isdisjoint(split.test_task_ids)
    assert len(set(split.explore_task_ids) | set(split.test_task_ids)) == 257
    assert split.task_ids("explore") == split.explore_task_ids
    assert split.task_ids("test") == split.test_task_ids


def test_frozen_split_rejects_a_changed_source_manifest(tmp_path: Path) -> None:
    changed = tmp_path / "manifest.json"
    changed.write_bytes(DEFAULT_FAMILY_MANIFEST.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest changed"):
        load_dabench_split(changed)


def test_react_skill_is_optional_and_only_changes_the_skill_prompt_section() -> None:
    context = {"goal_and_data": "analyze train.csv", "io_instructions": "write result.csv"}
    baseline = create_react_prompt(context)
    baseline_with_none = create_react_prompt(context, skill=None)
    skilled = create_react_prompt(context, skill="DATACOPE_SKILL_SENTINEL")

    assert baseline == baseline_with_none
    assert "Reusable Skill" not in baseline
    assert "DATACOPE_SKILL_SENTINEL" in skilled
    assert "task requirements and ReAct protocol take priority" in skilled
    assert AgentRuntimeConfig(skill_path="/tmp/SKILL.md").skill_path == "/tmp/SKILL.md"


def test_config_builder_accepts_the_optional_skill_path() -> None:
    config = ConfigBuilder().build_config(
        workflow="react",
        model="openai/test-model",
        api_key="test-key",
        agent_runtime={"max_steps": 3, "skill_path": "/tmp/SKILL.md"},
    )

    assert config.agent_runtime.max_steps == 3
    assert config.agent_runtime.skill_path == "/tmp/SKILL.md"

    with pytest.raises(ConfigurationError, match="skill_path.*non-empty"):
        ConfigBuilder().build_config(
            workflow="react",
            model="openai/test-model",
            api_key="test-key",
            agent_runtime={"skill_path": "  "},
        )


def test_react_factory_reads_the_configured_skill_file(tmp_path: Path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("DATACOPE_SKILL_SENTINEL\n", encoding="utf-8")
    config = type(
        "Config",
        (),
        {"agent_runtime": AgentRuntimeConfig(skill_path=str(skill_file))},
    )()

    assert _resolve_agent_skill(config) == "DATACOPE_SKILL_SENTINEL"


@pytest.mark.asyncio
async def test_react_workflow_puts_the_skill_only_in_the_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str | None] = {}
    react_operator = type("ReactOperator", (), {"max_steps": 1})()
    workflow = ReActWorkflow(
        operators={"react": react_operator, "execute": object()},
        services={"llm": object(), "sandbox": object(), "agent_skill": "SKILL_SENTINEL"},
        agent_config={},
    )

    async def fake_loop(  # noqa: ANN001, ANN202
        *,
        question,
        system_prompt,
        perception_system_prompt,
        output_path,
    ):
        captured["question"] = question
        captured["system_prompt"] = system_prompt
        captured["perception_system_prompt"] = perception_system_prompt
        return None, []

    monkeypatch.setattr(workflow, "_run_react_loop", fake_loop)
    await workflow.solve("task description", "write result.csv", tmp_path, tmp_path / "result.csv")

    assert isinstance(captured["system_prompt"], str)
    assert isinstance(captured["question"], str)
    assert "SKILL_SENTINEL" in captured["system_prompt"]
    assert "SKILL_SENTINEL" not in captured["question"]
    assert captured["perception_system_prompt"] is None


def test_run_spec_uses_frozen_phase_tasks_and_hashes_skill(tmp_path: Path) -> None:
    first_round = resolve_run_spec(phase="explore", run_root=tmp_path / "runs")
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: dabench\n---\n# DABench\n", encoding="utf-8")
    refined = resolve_run_spec(
        phase="explore",
        round_index=1,
        sample_index=9,
        run_root=tmp_path / "runs",
        skill_file=skill_file,
    )
    held_out = resolve_run_spec(
        phase="test",
        run_root=tmp_path / "runs",
        skill_file=skill_file,
    )

    assert PAPER_DISCOVERY_ROUNDS == 3
    assert PAPER_TRAJECTORIES_PER_TASK == 10
    assert PAPER_REACT_MAX_STEPS == 10
    assert len(first_round.task_ids) == 64
    assert first_round.workspace_root.name == "dabench_datacope_explore_round_0_sample_00"
    assert first_round.temperature == PAPER_EXPLORE_TEMPERATURE == 1.0
    assert refined.run_name == "dabench_datacope_explore_round_1_sample_09"
    assert refined.temperature == PAPER_EXPLORE_TEMPERATURE
    assert refined.skill_sha256 == hashlib.sha256(skill_file.read_bytes()).hexdigest()
    assert len(held_out.task_ids) == 193
    assert held_out.run_name == "dabench_datacope_test_skill"
    assert held_out.temperature == PAPER_EVALUATION_TEMPERATURE == 0.0

    with pytest.raises(ValueError, match="round 0 must run without"):
        resolve_run_spec(phase="explore", round_index=0, skill_file=skill_file)
    with pytest.raises(ValueError, match="round 1 requires"):
        resolve_run_spec(phase="explore", round_index=1)


def test_submission_signature_reports_missing_and_invalid(tmp_path: Path) -> None:
    missing = submission_signature(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    invalid = submission_signature(empty)

    assert missing == {
        "status": "missing",
        "prediction": "MISSING",
        "sha256": None,
        "rows": [],
    }
    assert invalid["status"] == "invalid"
    assert invalid["prediction"] == "INVALID"


def _write_task_workspace(
    root: Path,
    *,
    task_id: str,
    answer: str,
    score: float,
    workspace_name: str | None = None,
    ended_at: str = "2026-01-01T00:00:00Z",
) -> None:
    workspace = root / (workspace_name or task_id.replace("-", "_"))
    artifacts = workspace / "artifacts"
    telemetry = artifacts / "telemetry"
    telemetry.mkdir(parents=True)

    submission = root / f"submission_{workspace.name}.csv"
    submission.write_text(f"id,answer\n1,{answer}\n", encoding="utf-8")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "observation"},
        {"role": "assistant", "content": "answer"},
    ]
    (artifacts / "messages.json").write_text(json.dumps(messages), encoding="utf-8")
    metadata = {
        "task": {
            "task_id": task_id,
            "payload": {
                "description": f"Description for {task_id}",
                "output_submission_path": str(submission),
            },
        },
        "workspace_dir": str(workspace),
        "summary": {"result": {"submission_path": str(submission), "score": score}},
        "timeline": {"ended_at_utc": ended_at},
        "answers_path": "/private/ground_truth.csv",
    }
    (telemetry / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_export_workspace_run_emits_unsupervised_datacope_shape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    _write_task_workspace(
        workspace_root,
        task_id="dabench-2-second",
        answer="@value[2]",
        score=0.0,
    )
    _write_task_workspace(
        workspace_root,
        task_id="dabench-1-first",
        answer="@value[1]",
        score=1.0,
    )

    written = export_workspace_run(workspace_root, tmp_path / "predictions" / "run_0")
    payload = json.loads(written[0].read_text(encoding="utf-8"))

    assert len(written) == 2
    assert payload["extra_info"] == {"query_id": "dabench-1-first"}
    assert payload["prediction"].startswith("VALID:")
    assert payload["query"] == "Description for dabench-1-first"
    assert payload["turns"] == 2
    assert payload["submission"]["rows"] == [["id", "answer"], ["1", "@value[1]"]]
    assert "ground_truth" not in payload
    assert "score" not in payload
    assert "answers_path" not in payload


def test_export_workspace_run_keeps_only_latest_retry_without_using_score(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    _write_task_workspace(
        workspace_root,
        task_id="dabench-1-retried",
        answer="@value[old]",
        score=1.0,
        workspace_name="old_attempt",
        ended_at="2026-01-01T00:00:00Z",
    )
    _write_task_workspace(
        workspace_root,
        task_id="dabench-1-retried",
        answer="@value[new]",
        score=0.0,
        workspace_name="new_attempt",
        ended_at="2026-01-02T00:00:00Z",
    )

    written = export_workspace_run(workspace_root, tmp_path / "predictions")
    payload = json.loads(written[0].read_text(encoding="utf-8"))

    assert len(written) == 1
    assert payload["submission"]["rows"][-1][-1] == "@value[new]"
    assert "score" not in payload


def test_export_workspace_run_requires_react_messages(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    _write_task_workspace(
        workspace_root,
        task_id="dabench-1-no-messages",
        answer="@value[1]",
        score=1.0,
    )
    messages_file = next(workspace_root.glob("*/artifacts/messages.json"))
    messages_file.unlink()

    with pytest.raises(ValueError, match="Missing ReAct trajectory"):
        export_workspace_run(workspace_root, tmp_path / "predictions")


def test_prepare_public_data_excludes_private_files(tmp_path: Path) -> None:
    task = tmp_path / "source" / "dabench-1-example" / "prepared"
    public = task / "public"
    private = task / "private"
    public.mkdir(parents=True)
    private.mkdir()
    (public / "train.csv").write_text("value\n1\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text("answer\n\n", encoding="utf-8")
    (private / "answer.csv").write_text("answer\n1\n", encoding="utf-8")

    test_task = tmp_path / "source" / "dabench-2-held-out" / "prepared" / "public"
    test_task.mkdir(parents=True)
    (test_task / "train.csv").write_text("value\n2\n", encoding="utf-8")
    (test_task / "sample_submission.csv").write_text("answer\n\n", encoding="utf-8")

    written = prepare_public_data(
        tmp_path / "source",
        tmp_path / "public_view",
        ["dabench-1-example"],
    )

    assert written == [tmp_path / "public_view" / "dabench-1-example"]
    assert (written[0] / "train.csv").is_file()
    assert not list((tmp_path / "public_view").rglob("answer.csv"))
    assert not (tmp_path / "public_view" / "dabench-2-held-out").exists()


def test_prepare_public_data_rejects_empty_or_duplicate_task_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="At least one explore"):
        prepare_public_data(source, tmp_path / "empty", [])
    with pytest.raises(ValueError, match="must be unique"):
        prepare_public_data(source, tmp_path / "duplicate", ["dabench-1", "dabench-1"])


def test_exact_grouping_does_not_use_datacope_natural_language_similarity() -> None:
    predictions = [
        {"answer": "VALID:aaa256", "id": 1},
        {"answer": "VALID:bbb256", "id": 2},
        {"answer": "VALID:aaa256", "id": 3},
    ]

    groups = group_predictions_exactly(predictions)

    assert [[item["id"] for item in group] for group in groups] == [[1, 3], [2]]


def test_skill_output_contract_requires_the_exact_skill_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "round_0" / "dabench"

    prompt = skill_output_contract(skill_dir)

    assert prompt == (
        f"The final Skill file must be saved at exactly `{skill_dir.resolve() / 'SKILL.md'}`."
    )


def _write_prediction(run_dir: Path, task_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(run_dir.glob("prediction_*.json")))
    payload = {"prediction": "VALID:x", "extra_info": {"query_id": task_id}}
    (run_dir / f"prediction_{index}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_preflight_requires_matching_sample_runs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    for index in range(PAPER_TRAJECTORIES_PER_TASK):
        task_id = "dabench-2" if index == PAPER_TRAJECTORIES_PER_TASK - 1 else "dabench-1"
        _write_prediction(predictions / f"run_{index:02d}", task_id)

    with pytest.raises(ValueError, match="different task set"):
        validate_prediction_runs(predictions)


def test_prediction_preflight_requires_exactly_ten_runs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    for index in range(PAPER_TRAJECTORIES_PER_TASK - 1):
        _write_prediction(predictions / f"run_{index:02d}", "dabench-1")

    with pytest.raises(ValueError, match="exactly 10.*found 9"):
        validate_prediction_runs(predictions)


def test_prediction_preflight_accepts_ten_matching_runs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    for index in range(PAPER_TRAJECTORIES_PER_TASK):
        for task_id in ("dabench-1", "dabench-2"):
            _write_prediction(predictions / f"run_{index:02d}", task_id)

    assert validate_prediction_runs(predictions) == {"dabench-1", "dabench-2"}


def test_public_data_preflight_rejects_private_answers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    private = data_dir / "dabench-1" / "private"
    private.mkdir(parents=True)
    (private / "answer.csv").write_text("answer\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected public files"):
        validate_public_data(data_dir, {"dabench-1"})


def test_public_data_preflight_rejects_held_out_task_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for task_id in ("dabench-1", "dabench-2-held-out"):
        task_dir = data_dir / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "train.csv").write_text("value\n1\n", encoding="utf-8")
        (task_dir / "sample_submission.csv").write_text("answer\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra=.*dabench-2-held-out"):
        validate_public_data(data_dir, {"dabench-1"})


def test_sidecar_uses_requested_python_and_external_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    class Completed:
        returncode = 7

    def fake_run(command, check, env, timeout):  # noqa: ANN001
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        captured["timeout"] = timeout
        return Completed()

    monkeypatch.setattr("experiments.datacope_dabench.sidecar.subprocess.run", fake_run)
    returncode = run_sidecar(
        "verify",
        datacope_root=tmp_path / "datacope",
        predictions_dir=tmp_path / "predictions",
        verified_dir=tmp_path / "verified",
        python_executable=str(python),
    )

    command = captured["command"]
    assert returncode == 7
    assert isinstance(command, list)
    assert command[0] == str(python)
    assert command[2] == "verify"
    assert captured["check"] is False
    assert captured["env"] is None
    assert captured["timeout"] == 60 * 60


def test_discover_sidecar_defaults_to_codex_without_forwarding_endpoint_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(command, check, env, timeout):  # noqa: ANN001
        captured["command"] = command
        captured["env"] = env
        captured["timeout"] = timeout
        return Completed()

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
    codex_runtime_root = tmp_path / "codex-runtime"
    codex_bin = codex_runtime_root / "bin" / "codex"
    codex_bin.parent.mkdir(parents=True)
    codex_bin.write_text("", encoding="utf-8")
    for directory in (
        "datacope",
        "predictions",
        "predictions-round-0",
        "predictions-round-1",
        "previous-skill",
        "public",
    ):
        (tmp_path / directory).mkdir()
    (tmp_path / "previous-skill" / "SKILL.md").write_text(
        "# Previous\n", encoding="utf-8"
    )

    monkeypatch.setenv("API_KEY", "hkust-secret")
    monkeypatch.setenv("API_BASE", "https://gateway.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_CONFIGS", '{"contains":"secret"}')
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-forwarded")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        "experiments.datacope_dabench.sidecar._codex_runtime",
        lambda value: CodexRuntime(
            executable=codex_bin,
            root=codex_runtime_root,
            version="codex-cli 9.8.7",
            sha256="a" * 64,
        ),
    )
    monkeypatch.setattr("experiments.datacope_dabench.sidecar.subprocess.run", fake_run)
    returncode = run_sidecar(
        "discover",
        datacope_root=tmp_path / "datacope",
        predictions_dir=tmp_path / "predictions",
        verified_dir=tmp_path / "verified",
        data_dir=tmp_path / "public",
        skill_dir=tmp_path / "skill",
        round_index=2,
        previous_predictions_dirs=(
            tmp_path / "predictions-round-0",
            tmp_path / "predictions-round-1",
        ),
        previous_skill_dir=tmp_path / "previous-skill",
    )

    command = captured["command"]
    assert returncode == 0
    assert isinstance(command, list)
    environment = captured["env"]
    assert Path(command[0]).name == "bwrap"
    assert ["--tmpfs", "/"] == command[command.index("--tmpfs") : command.index("--tmpfs") + 2]
    assert not any("key" in argument.lower() for argument in command)
    assert isinstance(environment, dict)
    assert environment["CODEX_HOME"] == str(codex_home)
    assert environment["PATH"].startswith(str(codex_runtime_root / "codex-path"))
    assert "API_KEY" not in environment
    assert "API_BASE" not in environment
    assert "LLM_MODEL_CONFIGS" not in environment
    assert "UNRELATED_SECRET" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["timeout"] == 60 * 60

    ro_sources = {
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--ro-bind"
    }
    writable_sources = {
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--bind"
    }
    assert "/" not in ro_sources
    assert "/etc" not in ro_sources
    assert str((tmp_path / "public").resolve()) in ro_sources
    assert str((tmp_path / "predictions").resolve()) in ro_sources
    assert str((tmp_path / "predictions-round-0").resolve()) in ro_sources
    assert str((tmp_path / "predictions-round-1").resolve()) in ro_sources
    assert str((tmp_path / "previous-skill").resolve()) in ro_sources
    assert str((codex_home / "auth.json").resolve()) in ro_sources
    assert str(codex_runtime_root.resolve()) in ro_sources
    assert command[command.index("--codex-bin") + 1] == str(codex_bin)
    assert command[command.index("--codex-version") + 1] == "codex-cli 9.8.7"
    assert command[command.index("--codex-sha256") + 1] == "a" * 64
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert command[command.index("--round-index") + 1] == "2"
    previous_values = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--previous-predictions-dir"
    ]
    assert previous_values == [
        str((tmp_path / "predictions-round-0").resolve()),
        str((tmp_path / "predictions-round-1").resolve()),
    ]
    assert command[command.index("--previous-skill-dir") + 1] == str(
        (tmp_path / "previous-skill").resolve()
    )
    assert "--agent-type" not in command
    assert writable_sources == {
        str((tmp_path / "verified").resolve()),
        str((tmp_path / "skill").resolve()),
    }


def test_sidecar_timeout_is_reported_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, check, env, timeout):  # noqa: ANN001
        raise __import__("subprocess").TimeoutExpired(command, timeout)

    monkeypatch.setattr("experiments.datacope_dabench.sidecar.subprocess.run", fake_run)

    with pytest.raises(TimeoutError, match="verify timed out after 7 seconds"):
        run_sidecar(
            "verify",
            datacope_root=tmp_path / "datacope",
            predictions_dir=tmp_path / "predictions",
            verified_dir=tmp_path / "verified",
            timeout_seconds=7,
        )


def test_discovery_bubblewrap_hides_unmounted_held_out_data(tmp_path: Path) -> None:
    python = Path(sys.executable).absolute()
    datacope = tmp_path / "datacope"
    predictions = tmp_path / "predictions"
    public = tmp_path / "public"
    verified = tmp_path / "verified"
    skill = tmp_path / "skill"
    codex_home = tmp_path / "codex-home"
    codex_runtime = tmp_path / "codex-runtime"
    for directory in (
        datacope,
        predictions,
        public,
        verified,
        skill,
        codex_home,
        codex_runtime,
    ):
        directory.mkdir()

    public_file = public / "visible.txt"
    public_file.write_text("public", encoding="utf-8")
    held_out = tmp_path / "held-out-sentinel.txt"
    held_out.write_text("private", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    auth = codex_home / "auth.json"
    auth.write_text("{}", encoding="utf-8")

    probe = (
        "from pathlib import Path; "
        f"assert Path({str(public_file)!r}).read_text() == 'public'; "
        f"assert not Path({str(held_out)!r}).exists(); "
        f"Path({str(verified / 'probe.txt')!r}).write_text('ok'); "
        f"Path({str(skill / 'probe.txt')!r}).write_text('ok')"
    )
    command = _discovery_command(
        [str(python), "-c", probe],
        python_executable=python,
        datacope_root=datacope,
        predictions_dir=predictions,
        verified_dir=verified,
        data_dir=public,
        skill_dir=skill,
        family_manifest=manifest,
        auth_file=auth,
        codex_runtime_root=codex_runtime,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_discovery_environment(python, codex_home, codex_runtime),
    )
    if completed.returncode != 0 and "Operation not permitted" in completed.stderr:
        pytest.skip("The outer test sandbox forbids nested bubblewrap namespaces")

    assert completed.returncode == 0, completed.stderr
    assert (verified / "probe.txt").read_text(encoding="utf-8") == "ok"
    assert (skill / "probe.txt").read_text(encoding="utf-8") == "ok"


def test_codex_runtime_resolves_the_native_binary_from_an_npm_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "lib" / "node_modules" / "@openai" / "codex"
    javascript_entrypoint = package_root / "bin" / "codex.js"
    javascript_entrypoint.parent.mkdir(parents=True)
    javascript_entrypoint.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    runtime_root = (
        package_root
        / "node_modules"
        / "@openai"
        / "codex-linux-x64"
        / "vendor"
        / "x86_64-unknown-linux-musl"
    )
    native = runtime_root / "bin" / "codex"
    native.parent.mkdir(parents=True)
    native.write_text("native", encoding="utf-8")
    (runtime_root / "codex-package.json").write_text("{}", encoding="utf-8")

    entrypoint = tmp_path / "bin" / "codex"
    entrypoint.parent.mkdir()
    entrypoint.symlink_to(javascript_entrypoint)

    class Completed:
        returncode = 0
        stdout = "codex-cli 0.144.5\n"
        stderr = ""

    monkeypatch.setattr(
        "experiments.datacope_dabench.sidecar.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    runtime = _codex_runtime(str(entrypoint))

    assert runtime.executable == native.resolve()
    assert runtime.root == runtime_root.resolve()
    assert runtime.version == "codex-cli 0.144.5"
    assert runtime.sha256 == hashlib.sha256(b"native").hexdigest()


def test_datacope_patch_passes_the_selected_system_codex_binary() -> None:
    patch = (
        Path("experiments/datacope_dabench/patches/datacope-openai-codex.patch")
        .resolve()
        .read_text(encoding="utf-8")
    )

    assert 'raise ValueError("CodexAgent requires codex_bin")' in patch
    assert "codex_bin=str(self.codex_bin)" in patch
