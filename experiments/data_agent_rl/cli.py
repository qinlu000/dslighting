"""CLI for deterministic Data Agent RL Environment tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from dslighting.core.config.llm_resolution import build_llm_config

from .runner import (
    DEFAULT_BUCKET_ID,
    DETERMINISTIC_REWARD_MODES,
    BucketDataStore,
    RunSettings,
    load_tasks,
    run_tasks,
    select_tasks,
)

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
        description="Run deterministic Data Agent RL tasks with a DSLighting workflow"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bucket-cache-dir", type=Path)
    parser.add_argument("--bucket-id", default=DEFAULT_BUCKET_ID)
    parser.add_argument("--offline-data", action="store_true")
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", dest="task_ids", action="append", default=[])
    selection.add_argument(
        "--task-list-json",
        type=Path,
        help="JSON array of exact task IDs, such as the frozen 700-task selection",
    )
    selection.add_argument("--index", dest="index_expression")
    selection.add_argument("--all", dest="select_all", action="store_true")
    parser.add_argument(
        "--reward-mode",
        action="append",
        choices=sorted(DETERMINISTIC_REWARD_MODES),
        default=[],
        help="Restrict the strict deterministic subset; repeatable",
    )
    parser.add_argument("--difficulty", type=int, action="append", choices=range(0, 6), default=[])
    parser.add_argument("--package-tier", type=int, action="append", choices=range(0, 4), default=[])

    parser.add_argument("--workflow", choices=WORKFLOWS, default="react")
    parser.add_argument("--model")
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
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--download-concurrency", type=int, default=8)
    parser.add_argument("--prefetch-retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--compute-threads", type=int, default=2)
    parser.add_argument(
        "--local-isolation", choices=("process", "bubblewrap"), default="bubblewrap"
    )
    parser.add_argument(
        "--sandbox-python",
        type=Path,
        default=os.getenv("DATA_AGENT_RL_PYTHON"),
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
        dataset_root = args.dataset_root.expanduser().resolve()
        reward_modes = args.reward_mode or sorted(DETERMINISTIC_REWARD_MODES)
        tasks = load_tasks(dataset_root, reward_modes=reward_modes)
        if args.difficulty:
            allowed = set(args.difficulty)
            tasks = [task for task in tasks if task.difficulty_level in allowed]
        if args.package_tier:
            allowed = set(args.package_tier)
            tasks = [task for task in tasks if task.package_tier in allowed]
        task_ids = args.task_ids
        if args.task_list_json:
            task_ids = json.loads(args.task_list_json.expanduser().read_text(encoding="utf-8"))
            if not isinstance(task_ids, list) or not all(isinstance(value, str) for value in task_ids):
                raise ValueError("--task-list-json must contain a JSON array of task ID strings")
        selected = select_tasks(
            tasks,
            task_ids=task_ids,
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
        cache_dir = (
            args.bucket_cache_dir.expanduser().resolve()
            if args.bucket_cache_dir
            else dataset_root.parent / f"{dataset_root.name}_bucket_cache"
        )
        data_store = BucketDataStore(
            cache_root=cache_dir,
            bucket_id=args.bucket_id,
            token=os.getenv("HF_TOKEN") or False,
            offline=args.offline_data,
        )

        if args.prefetch_only:
            if min(args.download_concurrency, args.prefetch_retries) < 1:
                raise ValueError("Download concurrency and prefetch retries must be positive")
            grouped: dict[str, list] = {}
            for task in selected:
                grouped.setdefault(task.bucket_prefix, []).append(task)

            def ensure_group(tasks):
                for task in tasks:
                    data_store.ensure_dataset(task)

            pending = dict(grouped)
            errors: dict[str, str] = {}
            for attempt in range(1, args.prefetch_retries + 1):
                errors = {}
                with ThreadPoolExecutor(max_workers=args.download_concurrency) as pool:
                    futures = {
                        pool.submit(ensure_group, tasks): prefix
                        for prefix, tasks in pending.items()
                    }
                    for future in as_completed(futures):
                        prefix = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            errors[prefix] = f"{type(exc).__name__}: {exc}"
                pending = {
                    prefix: tasks
                    for prefix, tasks in pending.items()
                    if any(data_store.cached_dataset(task) is None for task in tasks)
                }
                print(
                    f"prefetch attempt={attempt}/{args.prefetch_retries} "
                    f"remaining_datasets={len(pending)}",
                    flush=True,
                )
                if not pending:
                    break
                if attempt < args.prefetch_retries:
                    time.sleep(min(2**attempt, 10))
            if pending:
                examples = {prefix: errors.get(prefix, "incomplete") for prefix in list(pending)[:10]}
                raise RuntimeError(
                    f"Prefetch incomplete for {len(pending)} datasets after "
                    f"{args.prefetch_retries} attempts: {examples}"
                )
            print(
                json.dumps(
                    {
                        "task_count": len(selected),
                        "unique_datasets": len({task.bucket_prefix for task in selected}),
                        "bucket_cache_dir": str(cache_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if not str(args.model or "").strip():
            raise ValueError("--model is required unless --prefetch-only is used")
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
                        "dataset_root": str(dataset_root),
                        "task_count": len(selected),
                        "reward_modes": {
                            mode: sum(task.reward_mode == mode for task in selected)
                            for mode in sorted({task.reward_mode for task in selected})
                        },
                        "difficulty_levels": {
                            str(level): sum(task.difficulty_level == level for task in selected)
                            for level in sorted({task.difficulty_level for task in selected})
                        },
                        "unique_datasets": len({task.bucket_prefix for task in selected}),
                        "workflow": args.workflow,
                        "model": llm_config.model,
                        "thinking": llm_config.thinking,
                        "concurrency": args.concurrency,
                        "download_concurrency": args.download_concurrency,
                        "max_steps": args.max_steps,
                        "sandbox": f"local/{args.local_isolation}",
                        "sandbox_python": str(args.sandbox_python or ""),
                        "compute_threads": args.compute_threads,
                        "sandbox_network_disabled": True,
                        "gold_exposed_to_agent": False,
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
            download_concurrency=args.download_concurrency,
            timeout_seconds=args.timeout_seconds,
            local_isolation=args.local_isolation,
            sandbox_python=args.sandbox_python,
            compute_threads=args.compute_threads,
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
