from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.datacope_agenticdatabench.adapter import (
    export_sample_run,
    output_signature,
    prepare_public_data,
    validate_public_data,
)
from experiments.datacope_agenticdatabench.datacope_bridge import (
    group_predictions_exactly,
    validate_prediction_runs,
)
from experiments.datacope_agenticdatabench.runner import evaluate_official, resolve_run_spec
from experiments.datacope_agenticdatabench.split import (
    create_split,
    load_split,
    load_task_records,
    required_output_names,
)


def _task_payload(index: int) -> dict:
    plot = index == 0
    return {
        "id": f"agriculture_{index:02d}",
        "question": f"Analyze public dataset {index}",
        "domain": f"domain_{index % 2}",
        "output_file_name": "figure.png" if plot else "output.csv",
        "post_process_func": ["image_post_process(figure.png)"] if plot else [],
        "gold_file": "hidden.csv",
        "eval_func": "hidden_evaluator",
        "skills": ["hidden benchmark label"],
    }


def _benchmark(tmp_path: Path, count: int = 8) -> tuple[Path, Path, Path]:
    benchmark = tmp_path / "AgenticDataBench"
    tasks_file = benchmark / "testbed" / "tasks" / "dev.jsonl"
    tasks_file.parent.mkdir(parents=True)
    tasks_file.write_text(
        "".join(json.dumps(_task_payload(index)) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    dataset_root = benchmark / "testbed" / "datasets"
    for index in range(2):
        domain = dataset_root / f"domain_{index}"
        domain.mkdir(parents=True)
        (domain / "data.csv").write_text("x\n1\n", encoding="utf-8")
    return benchmark, tasks_file, dataset_root


def _split(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    benchmark, tasks_file, dataset_root = _benchmark(tmp_path)
    manifest = tmp_path / "split.json"
    create_split(tasks_file, manifest)
    return benchmark, tasks_file, dataset_root, manifest


def test_split_is_deterministic_and_source_locked(tmp_path: Path) -> None:
    _, tasks_file, _, manifest = _split(tmp_path)
    split = load_split(manifest)
    split.verify_source(tasks_file)
    assert len(split.explore_task_ids) == 2
    assert len(split.test_task_ids) == 6
    assert not (set(split.explore_task_ids) & set(split.test_task_ids))

    tasks_file.write_text(
        tasks_file.read_text(encoding="utf-8") + json.dumps(_task_payload(9)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source changed"):
        split.verify_source(tasks_file)


def test_public_view_contains_only_safe_explore_fields(tmp_path: Path) -> None:
    _, tasks_file, dataset_root, manifest = _split(tmp_path)
    output = tmp_path / "public"
    prepare_public_data(tasks_file, dataset_root, manifest, output)
    split = load_split(manifest)
    roots = validate_public_data(output, split.query_ids("explore"))

    payload = json.loads((output / "tasks.json").read_text(encoding="utf-8"))
    rendered = json.dumps(payload)
    assert {item["query_id"] for item in payload} == split.query_ids("explore")
    assert "gold_file" not in rendered
    assert "eval_func" not in rendered
    assert "hidden benchmark label" not in rendered
    assert all(path.is_dir() for path in roots)


def _write_task_output(root: Path, task) -> None:
    task_dir = root / task.task_id
    task_dir.mkdir(parents=True)
    for name in required_output_names(task):
        (task_dir / name).write_bytes(f"{task.task_id}:{name}".encode())
    trajectory = [
        {"role": "system", "content": "DSLighting ReAct"},
        {"role": "user", "content": task.question},
        {"role": "assistant", "content": "<Think>inspect</Think><Action>code</Action>"},
        {"role": "user", "content": "<Observation>ok</Observation>"},
        {"role": "assistant", "content": "<Think>done</Think><Answer>done</Answer>"},
    ]
    result_path = task_dir / "dabench" / "result.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "finished": True,
                "trajectory": trajectory,
                "official_score": "must-not-leak",
                "eval_func": "must-not-leak",
            }
        ),
        encoding="utf-8",
    )


def test_export_uses_multifile_manifest_without_grader_data(tmp_path: Path) -> None:
    _, tasks_file, _, manifest = _split(tmp_path)
    split = load_split(manifest)
    tasks = {task.task_id: task for task in load_task_records(tasks_file)}
    run_output = tmp_path / "run"
    for task_id in split.explore_task_ids:
        _write_task_output(run_output, tasks[task_id])

    exported = tmp_path / "predictions" / "run_00"
    written = export_sample_run(run_output, tasks_file, manifest, exported)
    assert len(written) == len(split.explore_task_ids)
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    rendered = json.dumps(payload)
    assert payload["prediction"].startswith("VALID:")
    assert payload["submission"]["status"] == "valid"
    assert payload["turns"] == 2
    assert "must-not-leak" not in rendered
    assert "official_score" not in rendered
    assert "eval_func" not in rendered

    predictions = tmp_path / "predictions"
    for index in (1, 2):
        shutil.copytree(exported, predictions / f"run_{index:02d}")
    assert validate_prediction_runs(predictions) == split.query_ids("explore")


def test_output_signature_reports_missing_required_plot_sidecars(tmp_path: Path) -> None:
    task = load_task_records(
        _benchmark(tmp_path, count=2)[1]
    )[0]
    output = tmp_path / "output"
    output.mkdir()
    (output / "figure.png").write_bytes(b"png")
    signature = output_signature(task, output)
    assert signature["status"] == "missing"
    assert signature["missing_files"] == ["figure.json", "figure.npy"]


def test_exact_grouping_preserves_largest_first() -> None:
    predictions = [{"answer": "a"}, {"answer": "b"}, {"answer": "a"}]
    groups = group_predictions_exactly(predictions)
    assert [len(group) for group in groups] == [2, 1]


def test_runner_keeps_test_arms_separate(tmp_path: Path) -> None:
    benchmark, tasks_file, dataset_root, manifest = _split(tmp_path)
    run_root = tmp_path / "runs"
    baseline = resolve_run_spec(
        phase="test",
        benchmark_root=benchmark,
        tasks_file=tasks_file,
        dataset_root=dataset_root,
        split_manifest=manifest,
        run_root=run_root,
    )
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("# Skill\n", encoding="utf-8")
    skill = resolve_run_spec(
        phase="test",
        benchmark_root=benchmark,
        tasks_file=tasks_file,
        dataset_root=dataset_root,
        split_manifest=manifest,
        run_root=run_root,
        skill_file=skill_file,
    )
    assert baseline.output_dir != skill.output_dir
    assert baseline.temperature == skill.temperature == 0.0
    assert len(baseline.task_ids) == 6


def test_official_evaluator_uses_held_out_output(tmp_path: Path, monkeypatch) -> None:
    benchmark = tmp_path / "AgenticDataBench"
    testbed = benchmark / "testbed"
    evaluator = testbed / "evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    (testbed / "gold").mkdir()
    output = tmp_path / "test-output"
    output.mkdir()
    selected = output / "agenticdatabench_tasks.jsonl"
    selected.write_text("{}\n", encoding="utf-8")
    observed = {}

    def fake_run(command, *, cwd, check):
        observed.update(command=command, cwd=cwd, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "experiments.datacope_agenticdatabench.runner.subprocess.run",
        fake_run,
    )
    assert (
        evaluate_official(
            benchmark_root=benchmark,
            output_dir=output,
            python_executable="/venv/bin/python",
        )
        == 0
    )
    assert observed["command"][0] == "/venv/bin/python"
    assert observed["command"][-1] == str(selected.resolve())
    assert observed["cwd"] == testbed.resolve()
    assert observed["check"] is False
