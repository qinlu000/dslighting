"""Command-line entrypoint for the AgenticDataBench PoC sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from dslighting.core.config.llm_resolution import build_llm_config

from .runner import (
    DEFAULT_DATASETS_RELATIVE_PATH,
    DEFAULT_TASKS_RELATIVE_PATH,
    RunSettings,
    load_tasks,
    required_output_names,
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
        description="Run AgenticDataBench tasks with a DSLighting workflow"
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="AgenticDataBench checkout containing testbed/",
    )
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--dataset-root", type=Path)
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
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Explicitly enable provider reasoning mode",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Explicitly disable provider reasoning mode (default)",
    )
    parser.set_defaults(thinking=False)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=30,
        help="Maximum transport attempts for each LLM call (default: 30)",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--max-history-chars",
        type=int,
        default=48000,
        help="Maximum ReAct prompt-history characters (default: 48000)",
    )
    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=14,
        help="Number of recent ReAct turns retained verbatim (default: 14)",
    )
    parser.add_argument(
        "--summary-trigger-turns",
        type=int,
        default=18,
        help="Turn count that enables historical summarization (default: 18)",
    )
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--sandbox-backend",
        choices=("local", "docker", "e2b", "ds_sandbox"),
        default="local",
    )
    parser.add_argument("--docker-image")
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--cpu-cores", type=float, default=4.0)
    parser.add_argument("--pids-limit", type=int, default=256)
    parser.add_argument(
        "--local-isolation",
        choices=("process", "bubblewrap"),
        default="bubblewrap",
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--perception",
        action="store_true",
        help="Enable the Perception treatment (ReAct with offline Docker only)",
    )
    parser.add_argument(
        "--datacard-dir",
        type=Path,
        help="Add one validated, task-independent Datacard per visible domain root",
    )
    parser.add_argument("--skill-file", type=Path)
    parser.add_argument(
        "--llm-debug-logging",
        action="store_true",
        help="Enable verbose provider SDK logging to diagnose LLM retry causes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _enable_llm_debug_logging() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    for logger_name in (
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "litellm",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    benchmark_root = args.benchmark_root.expanduser().resolve()
    tasks_file = (
        args.tasks_file.expanduser().resolve()
        if args.tasks_file
        else benchmark_root / DEFAULT_TASKS_RELATIVE_PATH
    )
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else benchmark_root / DEFAULT_DATASETS_RELATIVE_PATH
    )
    output_dir = args.output_dir.expanduser().resolve()
    workspace_dir = (
        args.workspace_dir.expanduser().resolve()
        if args.workspace_dir
        else output_dir.parent / "workspaces"
    )
    staging_dir = (
        args.staging_dir.expanduser().resolve()
        if args.staging_dir
        else output_dir.parent / "staging"
    )
    return benchmark_root, tasks_file, dataset_root, output_dir, workspace_dir, staging_dir


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = _parser().parse_args(argv)
    try:
        (
            benchmark_root,
            tasks_file,
            dataset_root,
            output_dir,
            workspace_dir,
            staging_dir,
        ) = _resolve_paths(args)
        tasks = load_tasks(tasks_file)
        selected = select_tasks(
            tasks,
            task_ids=args.task_ids,
            index_expression=args.index_expression,
            select_all=args.select_all,
        )
        if args.sandbox_backend == "docker" and not str(args.docker_image or "").strip():
            raise ValueError("--sandbox-backend docker requires --docker-image")
        if args.concurrency <= 0:
            raise ValueError("--concurrency must be positive")
        if args.max_retries <= 0:
            raise ValueError("--max-retries must be positive")
        if args.max_history_chars <= 0:
            raise ValueError("--max-history-chars must be positive")
        if args.keep_recent_turns <= 0:
            raise ValueError("--keep-recent-turns must be positive")
        if args.summary_trigger_turns < args.keep_recent_turns:
            raise ValueError("--summary-trigger-turns must be >= --keep-recent-turns")
        if args.perception and (
            args.workflow != "react" or args.sandbox_backend != "docker" or args.allow_network
        ):
            raise ValueError(
                "--perception requires --workflow react, --sandbox-backend docker, "
                "and networking disabled"
            )
        if args.perception and args.datacard_dir is not None:
            raise ValueError("--perception and --datacard-dir are separate treatments")

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
            plan = {
                "benchmark_root": str(benchmark_root),
                "dataset_root": str(dataset_root),
                "output_dir": str(output_dir),
                "workflow": args.workflow,
                "model": llm_config.model,
                "thinking": llm_config.thinking,
                "concurrency": args.concurrency,
                "max_retries": args.max_retries,
                "max_steps": args.max_steps,
                "max_history_chars": args.max_history_chars,
                "keep_recent_turns": args.keep_recent_turns,
                "summary_trigger_turns": args.summary_trigger_turns,
                "timeout_seconds": args.timeout_seconds,
                "perception_enabled": args.perception,
                "datacard_enabled": args.datacard_dir is not None,
                "datacard_dir": (
                    str(args.datacard_dir.expanduser().resolve())
                    if args.datacard_dir is not None
                    else None
                ),
                "skill_file": (
                    str(args.skill_file.expanduser().resolve())
                    if args.skill_file is not None
                    else None
                ),
                "sandbox_backend": args.sandbox_backend,
                "docker_image": args.docker_image,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "domain": task.domain,
                        "required_outputs": list(required_output_names(task)),
                    }
                    for task in selected
                ],
            }
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        if args.llm_debug_logging:
            _enable_llm_debug_logging()

        settings = RunSettings(
            benchmark_root=benchmark_root,
            dataset_root=dataset_root,
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
            disable_network=not args.allow_network,
            perception_enabled=args.perception,
            datacard_dir=(
                args.datacard_dir.expanduser().resolve() if args.datacard_dir is not None else None
            ),
            skill_file=(
                args.skill_file.expanduser().resolve() if args.skill_file is not None else None
            ),
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
        )
        summary = asyncio.run(run_tasks(selected, settings))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
