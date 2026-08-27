"""Run local teacher rollouts on a selected training-mixture source."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from dslighting.config import LLMConfig
from experiments.common import read_jsonl

SOURCES = ("dare_bench", "data_agent_rl", "datamind_sql")


def _selected_ids(manifest: Path, source: str, limit: int | None) -> list[str]:
    task_ids = [str(row["task_id"]) for row in read_jsonl(manifest) if row.get("source") == source]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit_per_source must be positive")
        task_ids = task_ids[:limit]
    if not task_ids:
        raise ValueError(f"No {source} tasks selected from {manifest}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate {source} task IDs in {manifest}")
    return task_ids


def _clean_interrupted_outputs(output_dir: Path) -> int:
    if not output_dir.is_dir():
        return 0
    unfinished = [
        child
        for child in output_dir.iterdir()
        if child.is_dir() and not (child / "result.json").is_file()
    ]
    for path in unfinished:
        shutil.rmtree(path)
    return len(unfinished)


def _llm(args: argparse.Namespace) -> LLMConfig:
    return LLMConfig(
        model=args.model,
        provider="openai",
        api_base=args.api_base,
        api_key=args.api_key or os.getenv("API_KEY") or "local",
        temperature=args.temperature,
        thinking=False,
        max_retries=12,
        request_timeout_seconds=300,
        sdk_max_retries=0,
        max_concurrent_per_key=args.concurrency,
        global_max_concurrency=args.concurrency,
    )


async def run_source(args: argparse.Namespace) -> dict[str, Any]:
    task_ids = _selected_ids(
        args.selection_manifest or args.manifest,
        args.source,
        args.limit_per_source,
    )
    repetition = f"repeat_{args.repetition:02d}"
    output = args.run_root / "repetitions" / repetition / args.source
    workspace = args.run_root / "workspaces" / repetition / args.source
    staging = args.run_root / "staging" / args.source
    for path in (output, workspace, staging):
        path.mkdir(parents=True, exist_ok=True)
    removed = _clean_interrupted_outputs(output)
    if removed:
        print(f"removed_incomplete_outputs={removed}", flush=True)

    llm = _llm(args)
    common = {
        "output_dir": output,
        "workspace_dir": workspace,
        "staging_dir": staging,
        "workflow": "react",
        "llm": llm,
        "perception_enabled": args.perception,
        "perception_llm": llm if args.perception else None,
        "max_steps": args.max_steps,
        "protocol_mode": args.protocol_mode,
        "max_history_chars": 64_000,
        "keep_recent_turns": 12,
        "summary_trigger_turns": 16,
        "recent_observation_window": 12,
        "keep_latest_feedback_only": False,
        "max_feedback_retries": 1,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout_seconds,
        "local_isolation": "bubblewrap",
        "overwrite": args.overwrite,
        "retry_failed": args.retry_failed,
    }

    if args.source == "dare_bench":
        from experiments.dare_bench.runner import (
            DareDataStore,
            RunSettings,
            load_tasks,
            run_tasks,
            select_tasks,
        )

        tasks = select_tasks(load_tasks(args.dare_manifest), task_ids=task_ids)
        return await run_tasks(
            tasks,
            RunSettings(
                data_store=DareDataStore(databases_dir=args.dare_databases_dir),
                memory_mb=8192,
                cpu_cores=float(args.compute_threads),
                sandbox_python=args.dare_python,
                **common,
            ),
        )

    if args.source == "data_agent_rl":
        from experiments.data_agent_rl.runner import (
            BucketDataStore,
            RunSettings,
            load_tasks,
            run_tasks,
            select_tasks,
        )

        tasks = select_tasks(load_tasks(args.data_agent_root), task_ids=task_ids)
        return await run_tasks(
            tasks,
            RunSettings(
                data_store=BucketDataStore(
                    cache_root=args.data_agent_cache,
                    offline=True,
                ),
                download_concurrency=min(args.concurrency, 16),
                sandbox_python=args.general_python,
                compute_threads=args.compute_threads,
                **common,
            ),
        )

    from experiments.datamind_sql.runner import (
        DataMindSQLDataStore,
        RunSettings,
        load_tasks,
        run_tasks,
        select_tasks,
    )

    tasks = select_tasks(load_tasks(args.manifest), task_ids=task_ids)
    return await run_tasks(
        tasks,
        RunSettings(
            data_store=DataMindSQLDataStore(args.datamind_root),
            sandbox_python=args.general_python,
            compute_threads=args.compute_threads,
            **common,
        ),
    )


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _python_path(value: str) -> Path:
    """Keep a virtualenv interpreter path intact so Python detects its prefix."""
    return Path(value).expanduser().absolute()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-source",))
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--manifest", type=_path, required=True)
    parser.add_argument("--selection-manifest", type=_path)
    parser.add_argument("--dare-manifest", type=_path, required=True)
    parser.add_argument("--dare-databases-dir", type=_path, required=True)
    parser.add_argument("--data-agent-root", type=_path, required=True)
    parser.add_argument("--data-agent-cache", type=_path, required=True)
    parser.add_argument("--datamind-root", type=_path, required=True)
    parser.add_argument("--dare-python", type=_python_path, required=True)
    parser.add_argument("--general-python", type=_python_path, required=True)
    parser.add_argument("--run-root", type=_path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--compute-threads", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--protocol-mode",
        choices=("repair", "strict_retry"),
        default="repair",
    )
    parser.add_argument("--perception", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    args = build_parser().parse_args(argv)
    if min(args.concurrency, args.compute_threads, args.max_steps, args.timeout_seconds) <= 0:
        raise ValueError("Concurrency, threads, steps, and timeout must be positive")
    if args.repetition < 0:
        raise ValueError("repetition must be non-negative")
    value = asyncio.run(run_source(args))
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
