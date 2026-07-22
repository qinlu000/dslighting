from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.datacope_moscibench.adapter import (
    export_workspace_sample,
    normalize_answer,
    prepare_public_data,
    validate_public_data,
)
from experiments.datacope_moscibench.datacope_bridge import validate_prediction_runs
from experiments.datacope_moscibench.runner import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    resolve_run_spec,
    summarize_test,
)
from experiments.datacope_moscibench.split import DATASETS, create_split, load_split


def _fake_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "competitions"
    for family in DATASETS:
        for index in range(1, 5):
            task = root / f"mosci-{family}-{index}"
            public = task / "prepared" / "public"
            private = task / "prepared" / "private"
            public.mkdir(parents=True)
            private.mkdir()
            (task / "description.md").write_text(
                f"# {family} {index}\n\nPublic task description.\n",
                encoding="utf-8",
            )
            (public / "data.csv").write_text("x\n1\n", encoding="utf-8")
            (public / "sample_submission.csv").write_text(
                "id,answer\n1,@answer[value]\n",
                encoding="utf-8",
            )
            (private / "answer.csv").write_text("id,answer\n1,hidden\n", encoding="utf-8")
    return root


def _split(tmp_path: Path) -> tuple[Path, Path]:
    root = _fake_data_root(tmp_path)
    manifest = tmp_path / "split.json"
    create_split(root, manifest)
    return root, manifest


def test_split_is_family_stratified_and_source_locked(tmp_path: Path) -> None:
    root, manifest = _split(tmp_path)
    split = load_split(manifest)
    split.verify_source(root)
    assert all(len(split.explore[family]) == 1 for family in DATASETS)
    assert all(len(split.test[family]) == 3 for family in DATASETS)
    assert len(split.task_ids("explore")) == len(DATASETS)

    description = root / split.task_ids("explore")[0] / "description.md"
    description.write_text(description.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        split.verify_source(root)


def test_public_view_exposes_only_explore_public_data(tmp_path: Path) -> None:
    root, manifest = _split(tmp_path)
    output = tmp_path / "public"
    prepare_public_data(root, manifest, output)
    split = load_split(manifest)
    targets = validate_public_data(output, split.query_ids("explore"))

    tasks = json.loads((output / "tasks.json").read_text(encoding="utf-8"))
    assert {task["query_id"] for task in tasks} == split.query_ids("explore")
    assert len(targets) == len(DATASETS)
    assert all(path.name == "public" for path in targets)
    assert "hidden" not in json.dumps(tasks)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("@answer[1.00]", "number:1"),
        ("@answer[TRUE]", "bool:true"),
        ("@answer[ Random Forest ]", "text:random forest"),
        ("", "text:<missing>"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def _write_workspace_task(root: Path, task_id: str, answer: str = "@answer[1.00]") -> None:
    workspace = root / task_id.replace("-", "_")
    telemetry = workspace / "artifacts" / "telemetry"
    telemetry.mkdir(parents=True)
    submission = root / f"submission_{task_id}.csv"
    submission.write_text(f"id,answer\n1,{answer}\n", encoding="utf-8")
    messages = [
        {"role": "system", "content": "DSLighting ReAct"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<Think>inspect</Think><Action>code</Action>"},
        {"role": "user", "content": "<Observation>1</Observation>"},
        {"role": "assistant", "content": "<Think>done</Think><Answer>1</Answer>"},
    ]
    (workspace / "artifacts" / "messages.json").write_text(json.dumps(messages), encoding="utf-8")
    metadata = {
        "task": {
            "task_id": task_id,
            "payload": {
                "description": f"Description for {task_id}",
                "output_submission_path": str(submission),
            },
        },
        "workspace_dir": str(workspace),
        "summary": {"result": {"submission_path": str(submission), "score": 1.0}},
    }
    (telemetry / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_export_builds_unsupervised_datacope_conversations(tmp_path: Path) -> None:
    _, manifest = _split(tmp_path)
    split = load_split(manifest)
    workspace = tmp_path / "workspace"
    for task_id in split.task_ids("explore"):
        _write_workspace_task(workspace, task_id)

    output = tmp_path / "predictions"
    written = export_workspace_sample(workspace, manifest, output)
    assert len(written) == len(DATASETS)
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["prediction"] == "number:1"
    assert payload["extra_info"]["query_id"] in split.query_ids("explore")
    assert "score" not in json.dumps(payload)

    runs = tmp_path / "runs"
    for index in range(3):
        shutil.copytree(output, runs / f"run_{index:02d}")
    assert validate_prediction_runs(runs) == split.query_ids("explore")


def test_runner_uses_dslighting_data_and_separate_test_arms(tmp_path: Path) -> None:
    root, manifest = _split(tmp_path)
    run_root = tmp_path / "runs"
    baseline = resolve_run_spec(
        phase="test", data_root=root, split_manifest=manifest, run_root=run_root
    )
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("# Skill\n", encoding="utf-8")
    skill = resolve_run_spec(
        phase="test",
        data_root=root,
        split_manifest=manifest,
        run_root=run_root,
        skill_file=skill_file,
    )
    assert baseline.data_root == root.resolve()
    assert baseline.workspace_root != skill.workspace_root
    assert baseline.temperature == skill.temperature == 0.0
    assert len(baseline.task_ids) == 18
    assert DEFAULT_TASK_TIMEOUT_SECONDS == 7200

    with pytest.raises(ValueError, match="round 0"):
        resolve_run_spec(
            phase="explore",
            round_index=0,
            data_root=root,
            split_manifest=manifest,
            skill_file=skill_file,
        )


def test_summarize_test_reads_dslighting_run_manifests(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for arm, accuracy in (("baseline", 0.25), ("skill", 0.5)):
        path = run_root / "workspaces" / f"moscibench_datacope_test_{arm}" / "run_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "task_ids": ["mosci-cyclone-1", "mosci-terra-1"],
                    "accuracy": accuracy,
                    "datasets": {"cyclone": accuracy, "terra": accuracy},
                }
            ),
            encoding="utf-8",
        )
    summary = summarize_test(run_root)
    assert summary["baseline"]["accuracy"] == 0.25
    assert summary["skill"]["accuracy"] == 0.5
    assert summary["delta"] == 0.25
