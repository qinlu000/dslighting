"""CLI for running fixed-ground-truth DARE-Bench tasks with DSLighting."""

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

from .runner import DareDataStore, RunSettings, load_tasks, run_tasks, select_tasks

WORKFLOWS = (
    "react",
    "aide",
    "automind",
    "dsagent",
    "data_interpreter",
    "deepanalyze",
    "autokaggle",
    "aflow",
    "mini_swe_agent",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DARE-Bench fixed-ground-truth tasks with a DSLighting workflow"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--databases-zip", type=Path)
    data.add_argument("--databases-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", dest="task_ids", action="append", default=[])
    selection.add_argument("--index", dest="index_expression")
    selection.add_argument("--all", dest="select_all", action="store_true")
    parser.add_argument(
        "--task-type",
        action="append",
        choices=("classification", "regression", "time_series_analysis"),
        default=[],
        help="Filter the manifest before applying --index/--all; repeatable",
    )
    parser.add_argument(
        "--version",
        action="append",
        choices=("v1", "v2"),
        default=[],
        help="Filter the manifest before applying --index/--all; repeatable",
    )

    parser.add_argument("--workflow", choices=WORKFLOWS, default="react")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--api-base")
    parser.add_argument("--temperature", type=float, default=0.0)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=False)
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-history-chars", type=int, default=48000)
    parser.add_argument("--keep-recent-turns", type=int, default=14)
    parser.add_argument("--summary-trigger-turns", type=int, default=18)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--sandbox-backend",
        choices=("local", "docker", "e2b", "ds_sandbox"),
        default="local",
    )
    parser.add_argument("--docker-image")
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--cpu-cores", type=float, default=2.0)
    parser.add_argument("--pids-limit", type=int, default=256)
    parser.add_argument(
        "--local-isolation", choices=("process", "bubblewrap"), default="bubblewrap"
    )
    parser.add_argument(
        "--sandbox-python",
        type=Path,
        default=os.getenv("DARE_BENCH_PYTHON"),
        help=(
            "Python interpreter used only for local sandbox execution; defaults to "
            "$DARE_BENCH_PYTHON or the controller interpreter"
        ),
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = _parser().parse_args(argv)
    try:
        tasks = load_tasks(args.manifest)
        if args.task_type:
            allowed_types = set(args.task_type)
            tasks = [task for task in tasks if task.task_type in allowed_types]
        if args.version:
            allowed_versions = set(args.version)
            tasks = [task for task in tasks if task.version in allowed_versions]
        selected = select_tasks(
            tasks,
            task_ids=args.task_ids,
            index_expression=args.index_expression,
            select_all=args.select_all,
        )
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
        data_store = DareDataStore(
            databases_zip=args.databases_zip,
            databases_dir=args.databases_dir,
        )
        llm_config = build_llm_config(
            model=args.model,
            provider=args.provider,
            api_base=args.api_base,
            temperature=args.temperature,
            thinking=args.thinking,
            max_retries=args.max_retries,
            max_concurrent_per_key=args.concurrency,
            global_max_concurrency=args.concurrency,
        )

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "manifest": str(args.manifest.expanduser().resolve()),
                        "task_count": len(selected),
                        "task_types": {
                            task_type: sum(task.task_type == task_type for task in selected)
                            for task_type in sorted({task.task_type for task in selected})
                        },
                        "versions": {
                            version: sum(task.version == version for task in selected)
                            for version in sorted({task.version for task in selected})
                        },
                        "workflow": args.workflow,
                        "model": llm_config.model,
                        "concurrency": args.concurrency,
                        "max_steps": args.max_steps,
                        "sandbox": f"{args.sandbox_backend}/{args.local_isolation}",
                        "sandbox_python": str(args.sandbox_python or ""),
                        "ground_truth_exposed_to_agent": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        settings = RunSettings(
            data_store=data_store,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            staging_dir=staging_dir,
            workflow=args.workflow,
            llm=llm_config,
            max_steps=args.max_steps,
            max_history_chars=args.max_history_chars,
            keep_recent_turns=args.keep_recent_turns,
            summary_trigger_turns=args.summary_trigger_turns,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            sandbox_backend=args.sandbox_backend,
            docker_image=args.docker_image,
            memory_mb=args.memory_mb,
            cpu_cores=args.cpu_cores,
            pids_limit=args.pids_limit,
            local_isolation=args.local_isolation,
            sandbox_python=args.sandbox_python,
            disable_network=not args.allow_network,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
        )
        summary = asyncio.run(run_tasks(selected, settings))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
