"""CLI for AgenticDataBench × DataCOPE experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .adapter import export_sample_run, prepare_public_data
from .runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_RUN_ROOT,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    evaluate_official,
    resolve_run_spec,
    run_react,
)
from .sidecar import DEFAULT_SIDECAR_TIMEOUT_SECONDS, run_sidecar
from .split import (
    DEFAULT_DATASETS_RELATIVE_PATH,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_TASKS_RELATIVE_PATH,
    create_split,
)


def _tasks_file(benchmark_root: Path, value: Path | None) -> Path:
    return (
        value.expanduser().resolve()
        if value is not None
        else benchmark_root.expanduser().resolve() / DEFAULT_TASKS_RELATIVE_PATH
    )


def _dataset_root(benchmark_root: Path, value: Path | None) -> Path:
    return (
        value.expanduser().resolve()
        if value is not None
        else benchmark_root.expanduser().resolve() / DEFAULT_DATASETS_RELATIVE_PATH
    )


def _add_benchmark_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--dataset-root", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgenticDataBench to DataCOPE adapter")
    commands = parser.add_subparsers(dest="command", required=True)

    make_split = commands.add_parser("make-split")
    _add_benchmark_paths(make_split)
    make_split.add_argument("--output", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    make_split.add_argument("--explore-fraction", type=float, default=0.25)

    public = commands.add_parser("prepare-data")
    _add_benchmark_paths(public)
    public.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    public.add_argument("--output-dir", type=Path, required=True)

    run = commands.add_parser("run")
    _add_benchmark_paths(run)
    run.add_argument("--phase", choices=("explore", "test"), required=True)
    run.add_argument("--round-index", type=int)
    run.add_argument("--sample-index", type=int, default=0)
    run.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--skill-file", type=Path)
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TASK_TIMEOUT_SECONDS)
    run.add_argument(
        "--sandbox-backend",
        choices=("local", "docker"),
        default="docker",
    )
    run.add_argument("--docker-image")

    export = commands.add_parser("export")
    _add_benchmark_paths(export)
    export.add_argument("--run-output-dir", type=Path, required=True)
    export.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    export.add_argument("--output-dir", type=Path, required=True)

    discover = commands.add_parser("discover")
    discover.add_argument("--datacope-root", type=Path, required=True)
    discover.add_argument("--predictions-dir", type=Path, required=True)
    discover.add_argument("--verified-dir", type=Path, required=True)
    discover.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    discover.add_argument("--python", dest="python_executable", default=sys.executable)
    discover.add_argument("--timeout-seconds", type=int, default=DEFAULT_SIDECAR_TIMEOUT_SECONDS)
    discover.add_argument("--round-index", type=int, default=0)
    discover.add_argument(
        "--previous-predictions-dir",
        dest="previous_predictions_dirs",
        type=Path,
        action="append",
        default=[],
    )
    discover.add_argument("--previous-skill-dir", type=Path)
    discover.add_argument("--data-dir", type=Path, required=True)
    discover.add_argument("--skill-dir", type=Path, required=True)
    discover.add_argument("--model", default="gpt-5.5")
    discover.add_argument("--codex-bin", dest="codex_executable", default="codex")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--benchmark-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--python", dest="python_executable", default=sys.executable)
    evaluate.add_argument("--gold-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "make-split":
            benchmark = args.benchmark_root.expanduser().resolve()
            path = create_split(
                _tasks_file(benchmark, args.tasks_file),
                args.output,
                explore_fraction=args.explore_fraction,
            )
            print(f"Created frozen split: {path}")
            return 0

        if args.command == "prepare-data":
            benchmark = args.benchmark_root.expanduser().resolve()
            path = prepare_public_data(
                _tasks_file(benchmark, args.tasks_file),
                _dataset_root(benchmark, args.dataset_root),
                args.split_manifest,
                args.output_dir,
            )
            print(f"Created sanitized explore view: {path}")
            return 0

        if args.command == "run":
            spec = resolve_run_spec(
                phase=args.phase,
                benchmark_root=args.benchmark_root,
                tasks_file=args.tasks_file,
                dataset_root=args.dataset_root,
                split_manifest=args.split_manifest,
                run_root=args.run_root,
                round_index=args.round_index,
                sample_index=args.sample_index,
                model=args.model,
                skill_file=args.skill_file,
            )
            return run_react(
                spec,
                sandbox_backend=args.sandbox_backend,
                docker_image=args.docker_image,
                max_steps=args.max_steps,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout_seconds,
            )

        if args.command == "export":
            benchmark = args.benchmark_root.expanduser().resolve()
            written = export_sample_run(
                args.run_output_dir,
                _tasks_file(benchmark, args.tasks_file),
                args.split_manifest,
                args.output_dir,
            )
            print(f"Exported {len(written)} trajectories to {args.output_dir.resolve()}")
            return 0

        if args.command == "evaluate":
            return evaluate_official(
                benchmark_root=args.benchmark_root,
                output_dir=args.output_dir,
                python_executable=args.python_executable,
                gold_dir=args.gold_dir,
            )

        return run_sidecar(
            datacope_root=args.datacope_root,
            predictions_dir=args.predictions_dir,
            verified_dir=args.verified_dir,
            split_manifest=args.split_manifest,
            python_executable=args.python_executable,
            data_dir=args.data_dir,
            skill_dir=args.skill_dir,
            model=args.model,
            codex_executable=args.codex_executable,
            round_index=args.round_index,
            previous_predictions_dirs=args.previous_predictions_dirs,
            previous_skill_dir=args.previous_skill_dir,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
