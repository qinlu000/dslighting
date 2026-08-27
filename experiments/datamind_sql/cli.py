"""CLI for deterministic DataMind SQL tasks."""

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

from .runner import DataMindSQLDataStore, RunSettings, load_tasks, run_tasks, select_tasks

WORKFLOWS = ("react", "aide", "automind", "dsagent", "data_interpreter", "deepanalyze")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DataMind SQL tasks with DSLighting")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", dest="task_ids", action="append", default=[])
    selection.add_argument("--index", dest="index_expression")
    selection.add_argument("--all", dest="select_all", action="store_true")
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
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--compute-threads", type=int, default=2)
    parser.add_argument("--local-isolation", choices=("process", "bubblewrap"), default="bubblewrap")
    parser.add_argument(
        "--sandbox-python",
        type=Path,
        default=os.getenv("DATAMIND_SQL_PYTHON"),
        help="Python interpreter used only inside the local sandbox",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = _parser().parse_args(argv)
    try:
        tasks = load_tasks(args.manifest)
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
        store = DataMindSQLDataStore(args.dataset_root)
        llm = build_llm_config(
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
                        "dataset_root": str(store.dataset_root),
                        "task_count": len(selected),
                        "unique_databases": len({task.db_id for task in selected}),
                        "workflow": args.workflow,
                        "model": llm.model,
                        "thinking": llm.thinking,
                        "max_steps": args.max_steps,
                        "concurrency": args.concurrency,
                        "sandbox": f"local/{args.local_isolation}",
                        "sandbox_python": str(args.sandbox_python or ""),
                        "compute_threads": args.compute_threads,
                        "sandbox_network_disabled": True,
                        "ground_truth_exposed_to_agent": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        summary = asyncio.run(
            run_tasks(
                selected,
                RunSettings(
                    data_store=store,
                    output_dir=output_dir,
                    workspace_dir=workspace_dir,
                    staging_dir=staging_dir,
                    workflow=args.workflow,
                    llm=llm,
                    max_steps=args.max_steps,
                    concurrency=args.concurrency,
                    timeout_seconds=args.timeout_seconds,
                    local_isolation=args.local_isolation,
                    sandbox_python=args.sandbox_python,
                    compute_threads=args.compute_threads,
                    overwrite=args.overwrite,
                    retry_failed=args.retry_failed,
                ),
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
