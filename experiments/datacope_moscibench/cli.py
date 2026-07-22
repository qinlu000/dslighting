"""CLI for the MoSciBench × DataCOPE experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .adapter import export_workspace_sample, prepare_public_data
from .runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_RUN_ROOT,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    resolve_run_spec,
    run_react,
    summarize_test,
)
from .sidecar import run_sidecar
from .split import DEFAULT_SPLIT_MANIFEST, create_split


def _sidecar_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datacope-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=60 * 60)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoSciBench to DataCOPE adapter")
    commands = parser.add_subparsers(dest="command", required=True)

    make_split = commands.add_parser("make-split")
    make_split.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    make_split.add_argument("--output", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    make_split.add_argument("--explore-fraction", type=float, default=0.25)

    public = commands.add_parser("prepare-data")
    public.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    public.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    public.add_argument("--output-dir", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--phase", choices=("explore", "test"), required=True)
    run.add_argument("--round-index", type=int)
    run.add_argument("--sample-index", type=int, default=0)
    run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--skill-file", type=Path)
    run.add_argument("--max-concurrency", type=int, default=10)
    run.add_argument("--llm-max-concurrency", type=int, default=10)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TASK_TIMEOUT_SECONDS)

    export = commands.add_parser("export")
    export.add_argument("--workspace-root", type=Path, required=True)
    export.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    export.add_argument("--output-dir", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)

    discover = commands.add_parser("discover")
    _sidecar_args(discover)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "make-split":
            path = create_split(
                args.data_root,
                args.output,
                explore_fraction=args.explore_fraction,
            )
            print(f"Created frozen split: {path}")
            return 0
        if args.command == "prepare-data":
            path = prepare_public_data(args.data_root, args.split_manifest, args.output_dir)
            print(f"Created sanitized explore view: {path}")
            return 0
        if args.command == "export":
            written = export_workspace_sample(
                args.workspace_root,
                args.split_manifest,
                args.output_dir,
            )
            print(f"Exported {len(written)} trajectories to {args.output_dir.resolve()}")
            return 0
        if args.command == "summarize":
            print(summarize_test(args.run_root))
            return 0
        if args.command == "run":
            spec = resolve_run_spec(
                phase=args.phase,
                data_root=args.data_root,
                split_manifest=args.split_manifest,
                run_root=args.run_root,
                round_index=args.round_index,
                sample_index=args.sample_index,
                model=args.model,
                skill_file=args.skill_file,
            )
            return run_react(
                spec,
                max_concurrency=args.max_concurrency,
                llm_max_concurrency=args.llm_max_concurrency,
                timeout_seconds=args.timeout_seconds,
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
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
