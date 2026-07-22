"""Command line interface for the minimal DABench DataCOPE sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .adapter import export_workspace_run, prepare_public_data
from .datacope_bridge import validate_prediction_runs
from .protocol import PAPER_REACT_MAX_STEPS
from .runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_RUN_ROOT,
    resolve_run_spec,
    run_dabench_react,
)
from .sidecar import DEFAULT_CODEX_MODEL, DEFAULT_SIDECAR_TIMEOUT_SECONDS, run_sidecar
from .split import DEFAULT_FAMILY_MANIFEST, load_dabench_split


def _add_sidecar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datacope-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python from the DataCOPE environment",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SIDECAR_TIMEOUT_SECONDS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DABench to DataCOPE adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export one DSLighting sample run")
    export.add_argument("--workspace-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="run the frozen DABench split with ReAct")
    run.add_argument("--phase", choices=("explore", "test"), required=True)
    run.add_argument(
        "--round-index",
        type=int,
        help="zero-based discovery round (explore only; paper protocol uses 0, 1, 2)",
    )
    run.add_argument("--sample-index", type=int, default=0)
    run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run.add_argument("--family-manifest", type=Path, default=DEFAULT_FAMILY_MANIFEST)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--run-name")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--skill-file", type=Path)
    run.add_argument("--max-steps", type=int, default=PAPER_REACT_MAX_STEPS)
    run.add_argument("--max-concurrency", type=int, default=50)
    run.add_argument("--llm-max-concurrency", type=int, default=50)
    run.add_argument("--timeout-seconds", type=int, default=2 * 60 * 60)

    prepare_data = subparsers.add_parser(
        "prepare-data", help="copy a public-only DABench data view"
    )
    prepare_data.add_argument("--source-root", type=Path, required=True)
    prepare_data.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="exported explore runs whose exact task set will be copied",
    )
    prepare_data.add_argument(
        "--family-manifest",
        type=Path,
        default=DEFAULT_FAMILY_MANIFEST,
    )
    prepare_data.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="run agreement grouping without an LLM")
    _add_sidecar_arguments(verify)

    discover = subparsers.add_parser("discover", help="verify trajectories and create SKILL.md")
    _add_sidecar_arguments(discover)
    discover.add_argument("--round-index", type=int, default=0)
    discover.add_argument(
        "--previous-predictions-dir",
        dest="previous_predictions_dirs",
        type=Path,
        action="append",
        default=[],
        help="repeat in chronological order for every earlier discovery round",
    )
    discover.add_argument("--previous-skill-dir", type=Path)
    discover.add_argument("--data-dir", type=Path, required=True)
    discover.add_argument("--skill-dir", type=Path, required=True)
    discover.add_argument(
        "--family-manifest",
        type=Path,
        default=DEFAULT_FAMILY_MANIFEST,
    )
    discover.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help="Codex skill-writer model",
    )
    discover.add_argument(
        "--codex-bin",
        dest="codex_executable",
        default="codex",
        help="Updated Codex CLI entrypoint or native binary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            written = export_workspace_run(args.workspace_root, args.output_dir)
            print(f"Exported {len(written)} DABench trajectories to {args.output_dir.resolve()}")
            return 0
        if args.command == "run":
            spec = resolve_run_spec(
                phase=args.phase,
                sample_index=args.sample_index,
                round_index=args.round_index,
                data_root=args.data_root,
                family_manifest=args.family_manifest,
                run_root=args.run_root,
                run_name=args.run_name,
                model=args.model,
                skill_file=args.skill_file,
            )
            return run_dabench_react(
                spec,
                max_steps=args.max_steps,
                max_concurrency=args.max_concurrency,
                llm_max_concurrency=args.llm_max_concurrency,
                timeout_seconds=args.timeout_seconds,
            )
        if args.command == "prepare-data":
            task_ids = validate_prediction_runs(args.predictions_dir)
            expected = set(load_dabench_split(args.family_manifest).explore_task_ids)
            if task_ids != expected:
                raise ValueError(
                    "Prediction task set must equal the frozen explore split; "
                    f"missing={sorted(expected - task_ids)[:5]}, "
                    f"extra={sorted(task_ids - expected)[:5]}"
                )
            written = prepare_public_data(args.source_root, args.output_dir, task_ids)
            print(
                f"Copied public data for {len(written)} DABench tasks to {args.output_dir.resolve()}"
            )
            return 0

        return run_sidecar(
            args.command,
            datacope_root=args.datacope_root,
            predictions_dir=args.predictions_dir,
            verified_dir=args.verified_dir,
            python_executable=args.python_executable,
            data_dir=getattr(args, "data_dir", None),
            skill_dir=getattr(args, "skill_dir", None),
            family_manifest=getattr(args, "family_manifest", None),
            model=getattr(args, "model", None),
            codex_executable=getattr(args, "codex_executable", "codex"),
            round_index=getattr(args, "round_index", 0),
            previous_predictions_dirs=getattr(args, "previous_predictions_dirs", ()),
            previous_skill_dir=getattr(args, "previous_skill_dir", None),
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
