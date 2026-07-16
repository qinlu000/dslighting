"""Thin command line interface for versioned perception-skills protocols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .profile import (
    PROJECT_ROOT,
    ExperimentRequest,
    ExperimentSelection,
    ProtocolOverrides,
    load_profile,
)
from .runner import ExperimentConfigurationError, PerceptionSkillsRunner


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perception-skills")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="validate or execute one frozen perception-skills protocol",
    )
    run.add_argument("benchmark", choices=("dabench", "moscibench"))
    run.add_argument(
        "--execute",
        action="store_true",
        help="start paid downstream agents; otherwise validation-only",
    )
    run.add_argument("--repetitions", type=_positive_int, required=True)
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+")
    selection.add_argument("--tasks-file", type=Path)
    run.add_argument("--limit", type=_positive_int)
    run.add_argument("--resume", metavar="RUN_ID")
    run.add_argument("--allow-protocol-override", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--model", help=argparse.SUPPRESS)
    run.add_argument("--workflow", help=argparse.SUPPRESS)
    run.add_argument("--task-concurrency", type=_positive_int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = build_parser().parse_args(argv)
    try:
        profile = load_profile(args.benchmark)
        request = ExperimentRequest(
            execute=args.execute,
            repetitions=args.repetitions,
            selection=ExperimentSelection(
                tasks=tuple(args.tasks or ()),
                tasks_file=args.tasks_file,
                limit=args.limit,
            ),
            resume_run_id=args.resume,
            allow_protocol_override=args.allow_protocol_override,
            overrides=ProtocolOverrides(
                model=args.model,
                workflow=args.workflow,
                task_concurrency=args.task_concurrency,
            ),
        )
        manifest = PerceptionSkillsRunner(profile).run(request)
    except (ExperimentConfigurationError, RuntimeError, ValueError) as error:
        print(f"perception-skills: {error}", file=sys.stderr)
        return 2
    if manifest is not None:
        print(f"batch manifest: {manifest}")
    return 0
