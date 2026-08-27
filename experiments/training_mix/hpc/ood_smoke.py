"""Run one DABench and one AgenticDataBench task against a local model server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from dslighting.api import DSBenchmark
from dslighting.config import LLMConfig
from dslighting.core import ConfigBuilder
from experiments.agenticdatabench import runner as agentic
from experiments.common import cpu_thread_env


def _sandbox(python: Path, timeout_seconds: int) -> dict[str, Any]:
    return {
        "backend": "local",
        "timeout": timeout_seconds,
        "memory_mb": 8192,
        "cpu_cores": 2.0,
        "pids_limit": 256,
        "local_isolation": "bubblewrap",
        "environment_policy": "allowlist",
        "network_policy": "disabled",
        "python_executable": str(python),
    }


def _summarize_dabench_results(path: Path, expected_tasks: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_tasks:
        raise RuntimeError(
            f"DABench wrote {len(rows)} results for {expected_tasks} selected tasks: {path}"
        )
    try:
        scores = [float(row["score"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"DABench results contain an invalid score: {path}") from exc
    return {
        "tasks": len(rows),
        "average_score": sum(scores) / len(scores),
        "perfect_scores": sum(score == 1.0 for score in scores),
        "zero_scores": sum(score == 0.0 for score in scores),
    }


def run_dabench(args: argparse.Namespace, llm: LLMConfig) -> dict[str, Any]:
    output = args.output_root / "dabench"
    run_label = "dabench_ood_full" if args.full else "dabench_ood_smoke"
    config = ConfigBuilder().build_config(
        workflow="react",
        llm_config=llm,
        workspace_dir=str(output / "workspaces"),
        run_name=run_label,
        keep_workspace=not args.full,
        keep_workspace_on_failure=True,
        sandbox=_sandbox(args.dabench_python, args.timeout_seconds),
        data_analysis={"cache_enabled": False},
        agent_runtime={"max_steps": args.max_steps},
        output_contract={
            "require_output_before_completion": True,
            "missing_output_feedback_retries": 2,
        },
    )
    config.run.parameters = {
        **dict(config.run.parameters or {}),
        "sandbox_env": cpu_thread_env(2),
    }
    config.scheduler.max_concurrency = args.concurrency
    config.scheduler.cpu_worker_pool_size = args.concurrency
    project_root = Path(__file__).resolve().parents[3]
    benchmark = DSBenchmark(
        "dabench" if args.full else "custom",
        exp_name=run_label,
        data_dir=str(args.dabench_root),
        vendor_comp_dir=str(
            project_root / "dslighting" / "benchmark" / "vendor" / "dabench" / "competitions"
        ),
        competitions=None if args.full else [args.dabench_task],
    ).run(config=config, log_path=str(output / "results"), verbose=False)
    results_path = Path(benchmark.results_path)
    metadata_path = Path(benchmark.metadata_path)
    if not results_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("DABench did not write results and metadata")
    expected_tasks = 257 if args.full else 1
    return {
        "selection": "all_public" if args.full else args.dabench_task,
        **_summarize_dabench_results(results_path, expected_tasks),
        "results_path": str(results_path),
        "metadata_path": str(metadata_path),
    }


def run_agentic(args: argparse.Namespace, llm: LLMConfig) -> dict[str, Any]:
    testbed = args.agentic_root / "testbed"
    all_tasks = agentic.load_tasks(testbed / "tasks" / "dev.jsonl")
    tasks = agentic.select_tasks(
        all_tasks,
        task_ids=() if args.full else [args.agentic_task],
        select_all=args.full,
    )
    output = args.output_root / "agenticdatabench"
    summary = asyncio.run(
        agentic.run_tasks(
            tasks,
            agentic.RunSettings(
                benchmark_root=args.agentic_root,
                dataset_root=testbed / "datasets",
                output_dir=output,
                workspace_dir=output / "workspaces",
                staging_dir=output / "staging",
                workflow="react",
                llm=llm,
                max_steps=args.max_steps,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout_seconds,
                sandbox_backend="local",
                local_isolation="bubblewrap",
                sandbox_python=args.agentic_python,
                cpu_cores=2.0,
                disable_network=True,
            ),
        )
    )
    expected_tasks = len(tasks)
    if summary["completed"] + summary["failed"] != expected_tasks or summary["skipped"]:
        raise RuntimeError(
            f"AgenticDataBench did not account for {expected_tasks} rollouts: {summary}"
        )
    missing_shims = [
        task.task_id
        for task in tasks
        if not (output / task.task_id / "dabench" / "result.json").is_file()
    ]
    if missing_shims:
        raise RuntimeError(
            f"AgenticDataBench did not write evaluator shims for: {missing_shims[:10]}"
        )
    evaluator_log = output / "official_evaluator.log"
    evaluator_results = output / "official_results"
    completed = subprocess.run(
        [
            str(args.agentic_python),
            str(testbed / "evaluate.py"),
            "--output_dir",
            str(output),
            "--gold_dir",
            str(testbed / "gold"),
            "--eval_json",
            str(output / "agenticdatabench_tasks.jsonl"),
            "--result_dir",
            str(evaluator_results),
            "--timeout_seconds",
            str(args.evaluator_timeout_seconds),
        ],
        cwd=testbed,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
    )
    evaluator_log.write_text(
        completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"AgenticDataBench official evaluator failed: {evaluator_log}")
    score_path = evaluator_results / f"{output.name}.json"
    try:
        scores = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"AgenticDataBench official evaluator did not write valid scores: {score_path}"
        ) from exc
    if (
        scores.get("num_results") != expected_tasks
        or len(scores.get("results") or []) != expected_tasks
    ):
        raise RuntimeError(f"AgenticDataBench returned incomplete scores: {scores}")
    report = {
        "selection": "all_public" if args.full else args.agentic_task,
        "tasks": expected_tasks,
        "completed_rollouts": summary["completed"],
        "failed_rollouts": summary["failed"],
        "average_score": scores["average_score"],
        "average_finished": scores.get("average_finished"),
        "run_summary": str(output / "dslighting_run_summary.json"),
        "official_evaluator_log": str(evaluator_log),
        "official_scores": str(score_path),
    }
    if not args.full:
        report.update(
            {
                "rollout_status": summary["tasks"][0]["status"],
                "score": scores["results"][0]["total_score"],
            }
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dabench-root", type=Path, required=True)
    parser.add_argument("--dabench-python", type=Path, required=True)
    parser.add_argument("--agentic-root", type=Path, required=True)
    parser.add_argument("--agentic-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dabench-task", default="dabench-0-mean-fare-paid")
    parser.add_argument("--agentic-task", default="strategy_3")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all 257 DABench and all 246 public AgenticDataBench tasks",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--evaluator-timeout-seconds", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=False)
    llm = LLMConfig(
        model=args.model,
        provider="openai",
        api_base=args.api_base,
        api_key="dslighting-local",
        temperature=0.0,
        thinking=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_retries=3,
        request_timeout_seconds=300,
        sdk_max_retries=0,
        max_concurrent_per_key=args.concurrency,
        global_max_concurrency=args.concurrency,
    )
    report = {
        "mode": "full" if args.full else "smoke",
        "model": args.model,
        "api_base": args.api_base,
        "internal_thinking": False,
        "dabench": run_dabench(args, llm),
        "agenticdatabench": run_agentic(args, llm),
    }
    report_path = args.output_root / (
        "ood_full_summary.json" if args.full else "ood_smoke_summary.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ood_evaluation=passed mode={report['mode']} summary={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
