from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.datamind_perception.cli import main as cli_main
from experiments.datamind_perception.export_sft import export_sft
from experiments.datamind_perception.runner import (
    DataMindPerceptionTask,
    build_task_definition,
    load_tasks,
    prepare_agent_visible_dir,
    select_tasks,
    write_episode,
)


def _row(task_id: str, source_file: str) -> dict:
    question = f"Question: summarize {source_file}"
    return {
        "data_source": "darl/python",
        "db_id": None,
        "task_id": None,
        "trajectory": [],
        "prompt": [
            {
                "role": "system",
                "content": f"# Data Source\n**The data source path is 'data/files/{source_file}'.**",
            },
            {"role": "user", "content": question},
        ],
        "ability": "data-analysis",
        "reward_model": {
            "ground_truth": {"ground_truth": "42"},
            "style": "rule",
        },
        "extra_info": {"index": task_id, "question": question},
    }


def test_load_select_stage_and_build_definition(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "one.csv").write_text("x\n1\n", encoding="utf-8")
    (source_root / "two.csv").write_text("x\n2\n", encoding="utf-8")
    split = tmp_path / "train.parquet"
    pd.DataFrame([_row("task/one", "one.csv"), _row("task-two", "two.csv")]).to_parquet(
        split, index=False
    )

    tasks = load_tasks(split, source_root=source_root)
    assert [task.task_id for task in tasks] == ["task/one", "task-two"]
    assert tasks[0].ground_truth == "42"
    assert select_tasks(tasks, index_expression="0-1") == [tasks[0]]

    staged = prepare_agent_visible_dir(
        tasks[0], source_root=source_root, staging_root=tmp_path / "staging"
    )
    assert [path.name for path in staged.iterdir()] == ["one.csv"]
    assert (staged / "one.csv").resolve() == (source_root / "one.csv").resolve()

    definition = build_task_definition(
        tasks[0], agent_visible_dir=staged, output_dir=tmp_path / "output"
    )
    spec = definition.payload["execution_spec"]
    assert definition.mode == "open_ended"
    assert spec["description_text"] == (
        "# Data Source\n"
        "**The data source path is 'one.csv'.**\n\n"
        "Question: summarize one.csv"
    )
    assert spec["io_instructions"] == ""
    assert "42" not in spec["description_text"]
    assert "trajectory" not in json.dumps(spec)
    assert "submission_artifact_contract" not in spec


def test_export_sft_keeps_only_completed_perception_sessions(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episodes" / "task-one"
    episode_dir.mkdir(parents=True)
    messages = [
        {"role": "system", "content": "You are a Perception Agent."},
        {"role": "user", "content": "Exploration Request:\nInspect rows."},
        {
            "role": "assistant",
            "content": "<Action>```python\nprint('rows=1')\n```</Action>",
        },
        {"role": "user", "content": "<Observation>rows=1</Observation>"},
        {"role": "assistant", "content": "<Report>rows=1</Report>"},
    ]
    episode = {
        "episode_id": "task-one",
        "task_id": "task-one",
        "judge": {"score": 1.0},
        "perception_sessions": [
            {
                "agent_role": "perception",
                "segment_id": "perception-0001",
                "parent_solver_step": 1,
                "status": "completed",
                "messages": messages,
            },
            {
                "agent_role": "perception",
                "segment_id": "perception-0002",
                "status": "step_limit",
                "messages": messages[:-1],
            },
        ],
    }
    (episode_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
    output = tmp_path / "perception_sft.json"
    stats = export_sft(tmp_path / "episodes", output)

    assert stats["sessions_exported"] == 1
    records = json.loads(output.read_text(encoding="utf-8"))
    assert records[0]["messages"] == messages
    assert records[0]["metadata"]["agent_role"] == "perception"
    assert records[0]["metadata"]["final_reward"] == 1.0


def test_write_episode_extracts_solver_answer_instead_of_runner_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "messages.json").write_text(
        json.dumps(
            [
                {"role": "system", "content": "solver"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<Think>done</Think><Answer>42</Answer>"},
            ]
        ),
        encoding="utf-8",
    )
    (artifacts / "perception_sessions.json").write_text("[]", encoding="utf-8")
    task = DataMindPerceptionTask(
        task_id="task",
        source_task_id="task",
        question="question",
        source_file="data.csv",
        ground_truth="42",
        reward_style="rule",
        upstream_record={},
    )
    episode = write_episode(
        task,
        output_dir=tmp_path / "output",
        result=tmp_path / "runner-output",
        cost=0.0,
        usage={},
        record={"workspace_dir": str(workspace)},
    )
    assert episode["status"] == "completed"
    assert episode["answer"] == "42"
    assert episode["runner_result"] == str(tmp_path / "runner-output")


def test_cli_dry_run_supports_matched_no_perception_ablation(
    tmp_path: Path, capsys
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "one.csv").write_text("x\n1\n", encoding="utf-8")
    split = tmp_path / "benchmark.parquet"
    pd.DataFrame([_row("task-one", "one.csv")]).to_parquet(split, index=False)

    exit_code = cli_main(
        [
            "--split",
            str(split),
            "--source-root",
            str(source_root),
            "--output-dir",
            str(tmp_path / "output"),
            "--index",
            "0-1",
            "--model",
            "openai/DeepSeek-V4-Flash",
            "--no-perception",
            "--dry-run",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["workflow"] == "react"
    assert plan["perception_enabled"] is False
    assert plan["perception_model"] is None
