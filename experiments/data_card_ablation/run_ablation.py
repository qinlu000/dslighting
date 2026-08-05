#!/usr/bin/env python3
"""Run main/L1/L2/L3 task-context ablations on reviewed benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = "dabench"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
SUPPORTED_POLICIES = ("main", "l1", "l2", "l3")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dslighting.core.config.llm_resolution import build_llm_config  # noqa: E402
from experiments.data_card_ablation.engine import (  # noqa: E402
    ConditionExperimentEngine,
    ConditionRuntime,
    EngineRun,
    ExperimentCondition,
    ExperimentInvariantError,
    PreflightError,
    benchmark_manifest,
    build_configs,
    positive_int,
    preflight_l1,
    preflight_l2,
    prepare_benchmark,
    selection_manifest,
    utc_now,
)


def _policy_list(raw: str) -> tuple[str, ...]:
    policies = tuple(token.strip().lower() for token in raw.split(",") if token.strip())
    if not policies:
        raise argparse.ArgumentTypeError("expected at least one policy")
    invalid = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unsupported policies: " + ", ".join(invalid) + "; expected main,l1,l2,l3"
        )
    duplicates = sorted({policy for policy in policies if policies.count(policy) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError("duplicate policies: " + ", ".join(duplicates))
    return policies


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark task-context ablation: main, L1, L2, and L3."
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--data-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+")
    selection.add_argument("--tasks-file", type=Path)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--policies", type=_policy_list, default=SUPPORTED_POLICIES)
    parser.add_argument("--workflow")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--l1-artifact-dir", type=Path)
    parser.add_argument("--l2-guidance-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_id(source_id: str = DEFAULT_BENCHMARK) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in source_id)
    return f"{safe}_data_card_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _level_conditions(
    policies: Sequence[str],
    *,
    l1_artifact_dir: Path | None,
    l2_guidance_path: Path | None,
) -> tuple[ExperimentCondition, ...]:
    return tuple(
        ExperimentCondition(
            condition_id=policy,
            task_context_policy=policy,
            l1_artifact_dir=(l1_artifact_dir if policy in {"l1", "l3"} else None),
            l2_guidance_path=(l2_guidance_path if policy in {"l2", "l3"} else None),
        )
        for policy in policies
    )


def _execute(args: argparse.Namespace) -> Path | None:
    profile, prepared = prepare_benchmark(
        str(args.benchmark).strip(),
        args.data_root,
        inline_tasks=args.tasks,
        tasks_file=args.tasks_file,
        limit=args.limit,
    )
    policies = tuple(args.policies)
    l1_summary = None
    l1_dir = None
    if any(policy in {"l1", "l3"} for policy in policies):
        if args.l1_artifact_dir is None:
            raise PreflightError("--l1-artifact-dir is required by selected policy l1/l3")
        l1_summary = preflight_l1(args.l1_artifact_dir, prepared.targets)
        l1_dir = Path(l1_summary["directory"])
    l2_summary = None
    l2_path = None
    if any(policy in {"l2", "l3"} for policy in policies):
        l2_path, l2_summary = preflight_l2(args.l2_guidance_path or profile.l2_guidance_path)
    conditions = _level_conditions(
        policies,
        l1_artifact_dir=l1_dir,
        l2_guidance_path=l2_path,
    )
    run_id = _run_id(prepared.descriptor.source_id)
    runtime = ConditionRuntime(
        workflow=str(args.workflow or profile.default_workflow),
        llm=build_llm_config(model=args.model),
        task_concurrency=args.concurrency,
    )
    configs = build_configs(
        conditions=conditions,
        run_id=run_id,
        runtime=runtime,
        config_overrides=profile.config_overrides,
        dry_run=args.dry_run,
    )
    manifest: dict[str, Any] = {
        "schema_version": 4,
        "experiment": "data_card_ablation",
        "run_id": run_id,
        "status": "validated" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "created_at_utc": utc_now(),
        "benchmark": benchmark_manifest(prepared),
        "selection": selection_manifest(prepared),
        "runtime": {
            "workflow": runtime.workflow,
            "model": runtime.llm.model,
            "concurrency": runtime.task_concurrency,
            "output_contract": configs[conditions[0].condition_id].output_contract.model_dump(
                mode="json"
            ),
        },
        "conditions": [condition.as_manifest() for condition in conditions],
        "policies": list(policies),
        "artifacts": {
            key: value
            for key, value in (("l1", l1_summary), ("l2", l2_summary))
            if value is not None
        },
        "runs": [],
    }
    if args.dry_run:
        manifest["runs"] = [
            {"repetition": 1, **condition.as_manifest(), "status": "validated"}
            for condition in conditions
        ]
        manifest["completed_at_utc"] = utc_now()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return None
    run_root = (RUNS_ROOT / run_id).resolve()
    engine_run = EngineRun(
        repetition=1,
        run_id=run_id,
        configs=configs,
    )
    return ConditionExperimentEngine(run_root).execute(
        manifest=manifest,
        prepared=prepared,
        conditions=conditions,
        runs=(engine_run,),
        resume=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = _execute(args)
    except PreflightError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2
    except ExperimentInvariantError as error:
        print(f"experiment invariant failed: {error}", file=sys.stderr)
        return 3
    if manifest is not None:
        print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
