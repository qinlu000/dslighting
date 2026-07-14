import asyncio
from pathlib import Path
from types import SimpleNamespace

from dslighting.benchmark.benchmarks.mle_style_benchmark import MLEStyleBenchmark


def _benchmark(
    run_id: str | None,
    *,
    output_artifact_suffix_seed: str | None = None,
    log_path: Path | None = None,
) -> MLEStyleBenchmark:
    benchmark = object.__new__(MLEStyleBenchmark)
    benchmark.log_path = str(log_path or Path("/unused"))
    benchmark.runner = SimpleNamespace(
        config=SimpleNamespace(
            scheduler=SimpleNamespace(run_id=run_id),
            run=SimpleNamespace(
                parameters={
                    "output_artifact_suffix_seed": output_artifact_suffix_seed,
                }
            ),
        )
    )
    return benchmark


def test_output_suffix_is_stable_for_explicit_seed_across_mle_style_sources() -> None:
    first = _benchmark("scheduler-run-a", output_artifact_suffix_seed="ablation-run")
    second = _benchmark("scheduler-run-b", output_artifact_suffix_seed="ablation-run")

    for task_id in ("dabench-task", "mosci-pop_genetics-3", "mlebench-task"):
        assert first._output_artifact_suffix(task_id) == second._output_artifact_suffix(task_id)

    assert first._output_artifact_suffix("mosci-pop_genetics-3") != first._output_artifact_suffix(
        "mosci-pop_genetics-4"
    )


def test_output_suffix_retains_random_main_behavior_without_explicit_seed(
    monkeypatch,
) -> None:
    sentinel = SimpleNamespace(hex="abcdef123456")
    monkeypatch.setattr(
        "dslighting.benchmark.benchmarks.mle_style_benchmark.uuid.uuid4",
        lambda: sentinel,
    )

    assert _benchmark(None)._output_artifact_suffix("dabench-task") == "abcdef"
    assert _benchmark("shared-run")._output_artifact_suffix("mosci-pop_genetics-3") == "abcdef"


def test_seeded_output_retries_use_fresh_host_paths_with_one_solver_name(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark(
        "scheduler-run",
        output_artifact_suffix_seed="ablation-run",
        log_path=tmp_path / "main",
    )
    artifact_name = Path("submission_dabench-task_123456.csv")

    first = benchmark._allocate_output_artifact_path(artifact_name, seeded=True)
    first.write_text("stale result", encoding="utf-8")
    second = benchmark._allocate_output_artifact_path(artifact_name, seeded=True)

    assert first.name == second.name == artifact_name.name
    assert first.parent != second.parent
    assert first.exists()
    assert not second.exists()


def test_unseeded_output_keeps_main_host_path_shape(tmp_path: Path) -> None:
    benchmark = _benchmark("scheduler-run", log_path=tmp_path / "main")
    artifact_name = Path("submission_dabench-task_abcdef.csv")

    allocated = benchmark._allocate_output_artifact_path(artifact_name, seeded=False)

    assert allocated == (tmp_path / "main" / artifact_name).absolute()


def test_standard_mle_payload_carries_the_registry_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_dir = tmp_path / "task" / "prepared" / "public"
    public_dir.mkdir(parents=True)
    sample = public_dir / "sample_submission.csv"
    sample.write_text("value\n0\n", encoding="utf-8")
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    competition = SimpleNamespace(
        description="## Dataset Description\n\nPublic observations.",
        public_dir=public_dir,
        raw_dir=tmp_path / "raw",
        sample_submission=sample,
        submission_filename="sample_submission.csv",
        evaluator_config={},
        answers=tmp_path / "answer.csv",
    )
    registry = SimpleNamespace(
        get_competition=lambda _task_id: competition,
        get_competitions_dir=lambda: registry_dir,
    )
    submission = SimpleNamespace(to_payload=lambda: {"submission_artifact_contract": {}})
    benchmark = _benchmark(None, log_path=tmp_path / "logs")
    benchmark.name = "test"
    benchmark.data_source = "prepared"
    benchmark._get_registry_for_competition = lambda _task_id: registry
    benchmark._build_evaluation_contract = lambda **_kwargs: SimpleNamespace(
        grading=SimpleNamespace(submission=submission)
    )
    benchmark.log_mismatch = lambda **_kwargs: None
    monkeypatch.setattr(
        "dslighting.benchmark.benchmarks.mle_style_benchmark.is_dataset_prepared",
        lambda *_args, **_kwargs: True,
    )
    captured = []

    async def evaluate(task):
        captured.append(task)
        return "[ERROR] expected test stop", 0.0, {}

    asyncio.run(
        benchmark.evaluate_problem(
            {"competition_id": "mosci-test", "mode": "standard_ml"},
            evaluate,
        )
    )

    assert len(captured) == 1
    assert captured[0].payload["registry_dir"] == str(registry_dir)
