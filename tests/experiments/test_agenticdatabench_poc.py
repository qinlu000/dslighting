from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from dslighting.benchmark.grading.models import SubmissionArtifactContract
from dslighting.config import LLMConfig
from experiments.agenticdatabench_poc.cli import main
from experiments.agenticdatabench_poc.runner import (
    AgenticDataBenchTask,
    RunSettings,
    build_task_definition,
    load_tasks,
    prepare_agent_visible_dir,
    required_output_names,
    run_tasks,
    select_tasks,
    write_selected_task_config,
    write_upstream_result,
)


def _payload(task_id: str, *, plot: bool = False) -> dict:
    payload = {
        "id": task_id,
        "question": f"Solve {task_id} and save the requested files.",
        "domain": "agriculture",
        "data_sources": ["data.csv"],
        "skills": ["hidden solver label"],
        "output_file_name": ["output.png", "output.csv"] if plot else ["output.csv"],
        "gold_file_name": ["result.png", "result.csv"] if plot else ["result.csv"],
        "eval_func": ["must_not_reach_the_agent()"],
    }
    if plot:
        payload["post_process_func"] = ["image_post_process('output.png')"]
    return payload


def _write_tasks(path: Path) -> None:
    records = [_payload("agriculture_02"), _payload("agriculture_14", plot=True)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_select_and_plot_sidecars(tmp_path: Path) -> None:
    tasks_file = tmp_path / "dev.jsonl"
    _write_tasks(tasks_file)
    tasks = load_tasks(tasks_file)

    assert [task.task_id for task in select_tasks(tasks, task_ids=["agriculture_14"])] == [
        "agriculture_14"
    ]
    assert [task.task_id for task in select_tasks(tasks, index_expression="0-2")] == [
        "agriculture_02",
        "agriculture_14",
    ]
    assert required_output_names(tasks[1]) == (
        "output.png",
        "output.csv",
        "output.json",
        "output.npy",
    )

    with pytest.raises(ValueError, match="exactly one"):
        select_tasks(tasks)
    with pytest.raises(ValueError, match="Unknown task"):
        select_tasks(tasks, task_ids=["missing"])


def test_direct_execution_spec_excludes_gold_and_evaluator(tmp_path: Path) -> None:
    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output" / task.task_id

    definition = build_task_definition(
        task,
        agent_visible_dir=data_dir,
        output_dir=output_dir,
    )
    spec = definition.payload["execution_spec"]

    assert definition.task_type == "datasci"
    assert spec["output_path"] == str(output_dir.resolve())
    assert spec["submission_artifact_contract"]["root_kind"] == "directory"
    contract = SubmissionArtifactContract.from_payload(
        {"submission_artifact_contract": spec["submission_artifact_contract"]}
    )
    assert contract is not None
    assert contract.root_kind == "directory"
    assert contract.validation.required_children == ("output.csv",)
    serialized = json.dumps(definition.model_dump())
    assert "must_not_reach_the_agent" not in serialized
    assert "result.csv" not in serialized
    assert "hidden solver label" not in serialized


def test_selected_task_config_preserves_only_official_selected_records(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dev.jsonl"
    _write_tasks(source)
    destination = tmp_path / "run" / "agenticdatabench_tasks.jsonl"

    tasks = load_tasks(source)
    saved = write_selected_task_config([tasks[1]], destination)
    records = [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines()]

    assert [record["id"] for record in records] == ["agriculture_14"]
    assert records[0]["gold_file_name"] == ["result.png", "result.csv"]
    assert records[0]["eval_func"] == ["must_not_reach_the_agent()"]


def test_staging_links_domain_and_copies_plot_helper(tmp_path: Path) -> None:
    task = AgenticDataBenchTask.from_payload(_payload("agriculture_14", plot=True))
    dataset_root = tmp_path / "datasets"
    domain = dataset_root / "agriculture"
    domain.mkdir(parents=True)
    (domain / "data.csv").write_text("x\n1\n", encoding="utf-8")
    helper = tmp_path / "image.py"
    helper.write_text("class Plotprocess: pass\n", encoding="utf-8")

    stage = prepare_agent_visible_dir(
        task,
        dataset_root=dataset_root,
        staging_root=tmp_path / "staging",
        plot_helper=helper,
    )

    assert (stage / "data.csv").read_text(encoding="utf-8") == "x\n1\n"
    assert (stage / "data.csv").is_symlink()
    assert (stage / "image.py").read_text(encoding="utf-8") == helper.read_text(encoding="utf-8")
    assert not (stage / "image.py").is_symlink()


def test_staging_does_not_copy_dataset_when_symlinking_fails(tmp_path: Path, monkeypatch) -> None:
    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))
    dataset_root = tmp_path / "datasets"
    domain = dataset_root / "agriculture"
    domain.mkdir(parents=True)
    (domain / "large.csv").write_text("x\n1\n", encoding="utf-8")

    def reject_symlink(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(
        "experiments.agenticdatabench_poc.runner.os.symlink",
        reject_symlink,
    )

    with pytest.raises(OSError, match="symlinks unavailable"):
        prepare_agent_visible_dir(
            task,
            dataset_root=dataset_root,
            staging_root=tmp_path / "staging",
            plot_helper=tmp_path / "unused.py",
        )
    assert not (tmp_path / "staging" / task.task_id / "large.csv").exists()


def test_upstream_result_reports_missing_outputs_and_messages(tmp_path: Path) -> None:
    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))
    output_dir = tmp_path / "output" / task.task_id
    output_dir.mkdir(parents=True)
    (output_dir / "output.csv").write_text("value\n1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    telemetry = artifacts / "telemetry"
    telemetry.mkdir(parents=True)
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "done"},
    ]
    (artifacts / "messages.json").write_text(json.dumps(messages), encoding="utf-8")

    payload = write_upstream_result(
        task,
        output_dir=output_dir,
        result=output_dir,
        cost=0.25,
        usage={"total_tokens": 10},
        record={
            "task_id": task.task_id,
            "workspace_dir": str(workspace),
            "metadata_path": str(telemetry / "run_metadata.json"),
        },
    )

    assert payload["finished"] is True
    assert payload["steps"] == 1
    assert payload["trajectory"] == messages
    saved = json.loads((output_dir / "dabench" / "result.json").read_text(encoding="utf-8"))
    assert saved["dslighting"]["cost"] == 0.25


def test_upstream_result_reads_official_minisweagent_trajectory(
    tmp_path: Path,
) -> None:
    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))
    output_dir = tmp_path / "output" / task.task_id
    output_dir.mkdir(parents=True)
    (output_dir / "output.csv").write_text("value\n1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "use bash"},
        {"role": "exit", "content": "done"},
    ]
    (artifacts / "minisweagent_trajectory.json").write_text(
        json.dumps({"messages": messages}),
        encoding="utf-8",
    )

    payload = write_upstream_result(
        task,
        output_dir=output_dir,
        result=output_dir,
        cost=0.1,
        usage={},
        record={
            "task_id": task.task_id,
            "workspace_dir": str(workspace),
        },
    )

    assert payload["steps"] == 1
    assert payload["trajectory"] == messages


def test_cli_dry_run_is_explicit_and_does_not_require_datasets(tmp_path: Path, capsys) -> None:
    benchmark = tmp_path / "AgenticDataBench"
    tasks_file = benchmark / "testbed" / "tasks" / "dev.jsonl"
    _write_tasks(tasks_file)
    output_dir = tmp_path / "runs" / "dslighting-react-smoke"

    exit_code = main(
        [
            "--benchmark-root",
            str(benchmark),
            "--output-dir",
            str(output_dir),
            "--task",
            "agriculture_14",
            "--model",
            "test/model",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["tasks"][0]["required_outputs"] == [
        "output.png",
        "output.csv",
        "output.json",
        "output.npy",
    ]
    assert plan["concurrency"] == 30
    assert plan["thinking"] is False
    assert plan["max_retries"] == 30
    assert plan["max_steps"] == 30
    assert plan["max_history_chars"] == 48000
    assert plan["keep_recent_turns"] == 14
    assert plan["summary_trigger_turns"] == 18
    assert plan["timeout_seconds"] == 3600
    assert plan["perception_enabled"] is False
    assert not output_dir.exists()


def test_cli_accepts_minisweagent_with_benchmark_docker_image(
    tmp_path: Path,
    capsys,
) -> None:
    benchmark = tmp_path / "AgenticDataBench"
    tasks_file = benchmark / "testbed" / "tasks" / "dev.jsonl"
    _write_tasks(tasks_file)

    exit_code = main(
        [
            "--benchmark-root",
            str(benchmark),
            "--output-dir",
            str(tmp_path / "output"),
            "--task",
            "agriculture_02",
            "--workflow",
            "mini_swe_agent",
            "--model",
            "test/model",
            "--sandbox-backend",
            "docker",
            "--docker-image",
            "dslighting-agenticdatabench@sha256:abc",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow"] == "mini_swe_agent"
    assert plan["sandbox_backend"] == "docker"
    assert plan["docker_image"] == "dslighting-agenticdatabench@sha256:abc"


def test_cli_requires_image_for_docker_backend(tmp_path: Path, capsys) -> None:
    benchmark = tmp_path / "AgenticDataBench"
    tasks_file = benchmark / "testbed" / "tasks" / "dev.jsonl"
    _write_tasks(tasks_file)

    exit_code = main(
        [
            "--benchmark-root",
            str(benchmark),
            "--output-dir",
            str(tmp_path / "output"),
            "--task",
            "agriculture_02",
            "--model",
            "test/model",
            "--sandbox-backend",
            "docker",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--docker-image" in capsys.readouterr().err


def test_cli_perception_is_explicit_and_requires_offline_react_docker(
    tmp_path: Path, capsys
) -> None:
    benchmark = tmp_path / "AgenticDataBench"
    tasks_file = benchmark / "testbed" / "tasks" / "dev.jsonl"
    _write_tasks(tasks_file)
    common = [
        "--benchmark-root",
        str(benchmark),
        "--output-dir",
        str(tmp_path / "output"),
        "--task",
        "agriculture_02",
        "--model",
        "test/model",
        "--perception",
        "--dry-run",
    ]

    assert main(common) == 2
    assert "--perception requires" in capsys.readouterr().err

    assert (
        main(
            common[:-1]
            + [
                "--sandbox-backend",
                "docker",
                "--docker-image",
                "agenticdatabench:test",
                "--dry-run",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["perception_enabled"] is True


@pytest.mark.asyncio
async def test_run_tasks_writes_evaluator_layout_without_registry(
    tmp_path: Path, monkeypatch
) -> None:
    from dslighting.runner import DSLightingRunner

    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))
    benchmark = tmp_path / "AgenticDataBench"
    domain = benchmark / "testbed" / "datasets" / "agriculture"
    domain.mkdir(parents=True)
    (domain / "data.csv").write_text("x\n1\n", encoding="utf-8")

    def fake_get_eval_function(self):
        assert self.config.llm.thinking is False
        assert self.config.llm.max_retries == 30
        assert self.config.agent_runtime.perception_enabled is False
        assert self.config.agent_runtime.context.max_history_chars == 48000
        assert self.config.agent_runtime.context.keep_recent_turns == 14
        assert self.config.agent_runtime.context.summary_trigger_turns == 18
        assert self.config.agent_runtime.skill_path == str(skill_file.resolve())
        assert self.config.sandbox.backend == "docker"
        assert self.config.sandbox.docker_image == "agenticdatabench:test"
        assert self.config.sandbox.environment_policy == "allowlist"
        assert self.config.sandbox.network_policy == "disabled"

        async def evaluate(definition):
            spec = definition.payload["execution_spec"]
            output = Path(spec["output_path"])
            output.mkdir(parents=True)
            (output / "output.csv").write_text("value\n1\n", encoding="utf-8")
            return output, 0.1, {"total_tokens": 5}

        return evaluate

    monkeypatch.setattr(DSLightingRunner, "get_eval_function", fake_get_eval_function)
    output_root = tmp_path / "runs" / "dslighting-react-smoke"
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("# AgenticDataBench Skill\n", encoding="utf-8")
    settings = RunSettings(
        benchmark_root=benchmark,
        dataset_root=benchmark / "testbed" / "datasets",
        output_dir=output_root,
        workspace_dir=tmp_path / "runs" / "workspaces",
        staging_dir=tmp_path / "runs" / "staging",
        workflow="react",
        llm=LLMConfig(model="test/model", thinking=False, max_retries=30),
        sandbox_backend="docker",
        docker_image="agenticdatabench:test",
        disable_network=True,
        skill_file=skill_file,
    )

    summary = await run_tasks([task], settings)

    assert summary["completed"] == 1
    assert summary["thinking"] is False
    assert summary["max_history_chars"] == 48000
    assert summary["keep_recent_turns"] == 14
    assert summary["summary_trigger_turns"] == 18
    assert summary["skill_file"] == str(skill_file.resolve())
    result_path = output_root / task.task_id / "dabench" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["finished"] is True
    assert result["dslighting"]["cost"] == 0.1
    assert (output_root / task.task_id / "output.csv").is_file()
    selected_config = output_root / "agenticdatabench_tasks.jsonl"
    selected_records = [
        json.loads(line) for line in selected_config.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in selected_records] == ["agriculture_02"]


@pytest.mark.asyncio
async def test_run_tasks_fails_fast_on_adapter_errors(tmp_path: Path, monkeypatch) -> None:
    from dslighting.runner import DSLightingRunner

    task = AgenticDataBenchTask.from_payload(_payload("agriculture_02"))

    def fake_get_eval_function(self):
        async def evaluate(definition):
            raise AssertionError("evaluation should not start")

        return evaluate

    def reject_staging(*args, **kwargs):
        raise OSError("staging failed")

    monkeypatch.setattr(DSLightingRunner, "get_eval_function", fake_get_eval_function)
    monkeypatch.setattr(
        "experiments.agenticdatabench_poc.runner.prepare_agent_visible_dir",
        reject_staging,
    )
    output_root = tmp_path / "runs" / "output"
    settings = RunSettings(
        benchmark_root=tmp_path / "AgenticDataBench",
        dataset_root=tmp_path / "datasets",
        output_dir=output_root,
        workspace_dir=tmp_path / "runs" / "workspaces",
        staging_dir=tmp_path / "runs" / "staging",
        workflow="react",
        llm=LLMConfig(model="test/model"),
        local_isolation="process",
    )

    with pytest.raises(OSError, match="staging failed"):
        await run_tasks([task], settings)
    assert not (output_root / task.task_id / "dabench" / "result.json").exists()


@pytest.mark.asyncio
async def test_run_tasks_honors_task_concurrency(tmp_path: Path, monkeypatch) -> None:
    from dslighting.runner import DSLightingRunner

    payloads = [_payload(f"agriculture_{index:02d}") for index in range(3)]
    tasks = [AgenticDataBenchTask.from_payload(payload) for payload in payloads]
    benchmark = tmp_path / "AgenticDataBench"
    domain = benchmark / "testbed" / "datasets" / "agriculture"
    domain.mkdir(parents=True)
    (domain / "data.csv").write_text("x\n1\n", encoding="utf-8")
    active = 0
    max_active = 0

    def fake_get_eval_function(self):
        assert self.config.agent_runtime.perception_enabled is False

        async def evaluate(definition):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.03)
                output = Path(definition.payload["execution_spec"]["output_path"])
                output.mkdir(parents=True)
                (output / "output.csv").write_text("value\n1\n", encoding="utf-8")
                return output, 0.0, {}
            finally:
                active -= 1

        return evaluate

    monkeypatch.setattr(DSLightingRunner, "get_eval_function", fake_get_eval_function)
    output_root = tmp_path / "runs" / "output"
    settings = RunSettings(
        benchmark_root=benchmark,
        dataset_root=benchmark / "testbed" / "datasets",
        output_dir=output_root,
        workspace_dir=tmp_path / "runs" / "workspaces",
        staging_dir=tmp_path / "runs" / "staging",
        workflow="react",
        llm=LLMConfig(model="test/model"),
        concurrency=2,
        local_isolation="process",
    )

    summary = await run_tasks(tasks, settings)

    assert max_active == 2
    assert summary["concurrency"] == 2
    assert summary["completed"] == 3
    assert [item["task_id"] for item in summary["tasks"]] == [
        "agriculture_00",
        "agriculture_01",
        "agriculture_02",
    ]
