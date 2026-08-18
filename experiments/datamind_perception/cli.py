"""Collect DeepSeek teacher trajectories for DataMind FastPerception."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from dslighting.core.config.llm_resolution import build_llm_config

from .runner import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLIT,
    RunSettings,
    load_tasks,
    run_tasks,
    select_tasks,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", dest="task_ids", action="append", default=[])
    selection.add_argument("--index", dest="index_expression")
    selection.add_argument("--all", dest="select_all", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--temperature", type=float, default=0.0)
    perception = parser.add_mutually_exclusive_group()
    perception.add_argument(
        "--perception",
        dest="perception_enabled",
        action="store_true",
        help="Enable FastPerception (default)",
    )
    perception.add_argument(
        "--no-perception",
        dest="perception_enabled",
        action="store_false",
        help="Run the matched ReAct baseline without <Explore>",
    )
    parser.set_defaults(perception_enabled=True)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-history-chars", type=int, default=48000)
    parser.add_argument("--keep-recent-turns", type=int, default=14)
    parser.add_argument("--summary-trigger-turns", type=int, default=18)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--cpu-cores", type=float, default=4.0)
    parser.add_argument("--pids-limit", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)
    args = _parser().parse_args(argv)
    try:
        model = str(args.model or os.getenv("LLM_MODEL") or "").strip()
        api_base = str(args.api_base or os.getenv("API_BASE") or "").strip() or None
        if not model:
            raise ValueError("Teacher model is required via --model or LLM_MODEL")
        output_dir = args.output_dir.expanduser().resolve()
        workspace_dir = (
            args.workspace_dir.expanduser().resolve()
            if args.workspace_dir
            else output_dir.parent / f"{output_dir.name}_workspaces"
        )
        staging_dir = (
            args.staging_dir.expanduser().resolve()
            if args.staging_dir
            else output_dir.parent / f"{output_dir.name}_staging"
        )
        tasks = load_tasks(args.split, source_root=args.source_root)
        selected = select_tasks(
            tasks,
            task_ids=args.task_ids,
            index_expression=args.index_expression,
            select_all=args.select_all,
        )
        llm = build_llm_config(
            model=model,
            api_base=api_base,
            temperature=args.temperature,
            thinking=False,
            max_retries=args.max_retries,
            max_concurrent_per_key=args.concurrency,
            global_max_concurrency=args.concurrency,
        )
        plan = {
            "split": str(args.split.expanduser().resolve()),
            "source_root": str(args.source_root.expanduser().resolve()),
            "output_dir": str(output_dir),
            "workflow": "fastperception" if args.perception_enabled else "react",
            "solver_model": llm.model,
            "perception_model": llm.model if args.perception_enabled else None,
            "perception_enabled": args.perception_enabled,
            "sandbox": "local/bubblewrap/offline",
            "task_count": len(selected),
            "task_ids": [task.task_id for task in selected],
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        settings = RunSettings(
            source_root=args.source_root,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            staging_dir=staging_dir,
            llm=llm,
            perception_enabled=args.perception_enabled,
            max_steps=args.max_steps,
            max_history_chars=args.max_history_chars,
            keep_recent_turns=args.keep_recent_turns,
            summary_trigger_turns=args.summary_trigger_turns,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            memory_mb=args.memory_mb,
            cpu_cores=args.cpu_cores,
            pids_limit=args.pids_limit,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
        )
        summary = asyncio.run(run_tasks(selected, settings))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
