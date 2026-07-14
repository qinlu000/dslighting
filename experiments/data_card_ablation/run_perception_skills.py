#!/usr/bin/env python3
"""Measure perception-skill effects through downstream benchmark performance."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = "moscibench"
DEFAULT_SKILLS_MANIFEST = EXPERIMENT_ROOT / "skills" / "perception_skills.json"
DEFAULT_ANNOTATIONS_ROOT = EXPERIMENT_ROOT / "artifacts" / "annotations"
RUNS_ROOT = EXPERIMENT_ROOT / "runs" / "perception_skills"
MAIN_REFERENCE_ID = "main"
MAIN_TASK_CONTEXT_POLICY = "main"
SKILL_TASK_CONTEXT_POLICY = "l1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_card_ablation import run_ablation as level_runtime  # noqa: E402

PreflightError = level_runtime.PreflightError
ExperimentInvariantError = level_runtime.ExperimentInvariantError


@dataclass(frozen=True)
class PerceptionSkillCondition:
    """One fixed annotation condition in the downstream experiment."""

    perception_skill_id: str
    skill_entrypoints: tuple[Path, ...]
    annotations_dir: Path

    @property
    def is_control(self) -> bool:
        return not self.skill_entrypoints


@dataclass(frozen=True)
class DownstreamCondition:
    """One condition presented to the fixed downstream Solving Agent."""

    condition_id: str
    condition_type: str
    task_context_policy: str
    perception_skill: PerceptionSkillCondition | None = None

    @property
    def annotations_dir(self) -> Path | None:
        if self.perception_skill is None:
            return None
        return self.perception_skill.annotations_dir

    @property
    def perception_skill_id(self) -> str | None:
        if self.perception_skill is None:
            return None
        return self.perception_skill.perception_skill_id

    @property
    def is_control(self) -> bool:
        return bool(self.perception_skill and self.perception_skill.is_control)


def _perception_skill_list(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one perception skill ID")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            "duplicate perception skill IDs: " + ", ".join(duplicates)
        )
    invalid = [value for value in values if not _is_safe_id(value)]
    if invalid:
        raise argparse.ArgumentTypeError(
            "perception skill IDs must be safe basenames: " + ", ".join(invalid)
        )
    return values


def _is_safe_id(value: str) -> bool:
    return bool(
        value
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed downstream tasks with L1 annotations produced under "
            "different perception skills."
        )
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Reviewed MLE-style benchmark source (default: {DEFAULT_BENCHMARK}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Benchmark data root containing prepared task directories.",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--tasks", nargs="+", help="Task IDs separated by spaces or commas.")
    task_group.add_argument(
        "--tasks-file",
        type=Path,
        help="UTF-8 file with one task ID per line; comments start with #.",
    )
    parser.add_argument("--limit", type=level_runtime._positive_int)
    parser.add_argument(
        "--perception-skills",
        type=_perception_skill_list,
        help=(
            "Comma-separated perception skill IDs. By default every condition "
            "in the manifest is selected."
        ),
    )
    parser.add_argument(
        "--exclude-main",
        action="store_true",
        help="Exclude the benchmark Dataset Description reference condition.",
    )
    parser.add_argument(
        "--perception-skills-manifest",
        type=Path,
        default=DEFAULT_SKILLS_MANIFEST,
        help="Manifest describing the perception skills and control condition.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=DEFAULT_ANNOTATIONS_ROOT,
        help="Root containing <perception_skill_id>/<dataset_id>.json.",
    )
    parser.add_argument(
        "--workflow",
        help="Override the benchmark profile's fixed downstream workflow.",
    )
    parser.add_argument(
        "--model",
        help="Downstream model override; defaults to LLM_MODEL from .env/environment.",
    )
    parser.add_argument(
        "--concurrency",
        type=level_runtime._positive_int,
        default=1,
        help="Maximum concurrent benchmark tasks (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the complete experiment and print its manifest without running agents.",
    )
    return parser


def _strict_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    payload = level_runtime._read_bytes(path, label)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PreflightError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be a JSON object: {path}")
    return value, payload


def _resolve_child(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PreflightError(f"{label} escapes its declared root: {resolved}") from error
    return resolved


def _load_perception_skills(
    manifest_path: Path,
    annotations_root: Path,
    selected_ids: Sequence[str] | None,
) -> tuple[tuple[PerceptionSkillCondition, ...], dict[str, Any]]:
    manifest = manifest_path.expanduser().resolve()
    document, payload = _strict_json(manifest, "perception skills manifest")
    if document.get("schema_version") != "perception_skills_v1":
        raise PreflightError(
            "perception skills manifest schema_version must be 'perception_skills_v1'"
        )
    raw_conditions = document.get("perception_skills")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise PreflightError("perception skills manifest must contain a non-empty list")

    annotations = level_runtime._resolve_directory(annotations_root, "annotations root")
    skills_root = manifest.parent.resolve()
    by_id: dict[str, PerceptionSkillCondition] = {}
    entrypoint_summaries: dict[str, list[dict[str, str]]] = {}
    for index, raw in enumerate(raw_conditions):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"perception_skills[{index}] must be an object")
        unexpected = set(raw) - {"perception_skill_id", "skill_entrypoints"}
        if unexpected:
            raise PreflightError(
                f"perception_skills[{index}] has unsupported fields: "
                + ", ".join(sorted(unexpected))
            )
        skill_id = raw.get("perception_skill_id")
        if not isinstance(skill_id, str) or not _is_safe_id(skill_id):
            raise PreflightError(
                f"perception_skills[{index}].perception_skill_id must be a safe basename"
            )
        if skill_id in by_id:
            raise PreflightError(f"duplicate perception skill ID in manifest: {skill_id}")
        raw_entrypoints = raw.get("skill_entrypoints")
        if not isinstance(raw_entrypoints, list) or not all(
            isinstance(value, str) and value for value in raw_entrypoints
        ):
            raise PreflightError(
                f"perception skill {skill_id!r} must provide a skill_entrypoints list"
            )

        entrypoints: list[Path] = []
        summaries: list[dict[str, str]] = []
        for relative in raw_entrypoints:
            candidate = Path(relative)
            if candidate.is_absolute():
                raise PreflightError(
                    f"perception skill {skill_id!r} entrypoint must be relative: {relative}"
                )
            entrypoint = _resolve_child(
                skills_root / candidate,
                skills_root,
                f"perception skill {skill_id!r} entrypoint",
            )
            entrypoint_payload = level_runtime._read_bytes(
                entrypoint, f"perception skill {skill_id!r} entrypoint"
            )
            entrypoints.append(entrypoint)
            summaries.append(
                {
                    "path": str(entrypoint),
                    "sha256": level_runtime._sha256(entrypoint_payload),
                }
            )

        annotations_dir = _resolve_child(
            annotations / skill_id,
            annotations,
            f"perception skill {skill_id!r} annotations",
        )
        annotations_dir = level_runtime._resolve_directory(
            annotations_dir, f"perception skill {skill_id!r} annotations"
        )
        condition = PerceptionSkillCondition(
            perception_skill_id=skill_id,
            skill_entrypoints=tuple(entrypoints),
            annotations_dir=annotations_dir,
        )
        by_id[skill_id] = condition
        entrypoint_summaries[skill_id] = summaries

    requested = tuple(selected_ids) if selected_ids is not None else tuple(by_id)
    unknown = [skill_id for skill_id in requested if skill_id not in by_id]
    if unknown:
        raise PreflightError(
            "unknown perception skill IDs: " + ", ".join(unknown)
        )
    selected = tuple(by_id[skill_id] for skill_id in requested)
    if not selected:
        raise PreflightError("perception skill selection is empty")

    summary = {
        "path": str(manifest),
        "sha256": level_runtime._sha256(payload),
        "schema_version": document["schema_version"],
        "selected": [
            {
                "perception_skill_id": condition.perception_skill_id,
                "control": condition.is_control,
                "skill_entrypoints": entrypoint_summaries[condition.perception_skill_id],
                "annotations_dir": str(condition.annotations_dir),
            }
            for condition in selected
        ],
    }
    return selected, summary


def _fixed_config_fingerprint(config: Any) -> str:
    payload = config.model_dump(mode="json")
    run = payload.get("run")
    if isinstance(run, dict):
        run.pop("run_name", None)
    task_context = payload.get("task_context")
    if isinstance(task_context, dict):
        task_context.pop("policy", None)
        task_context.pop("l1_artifact_dir", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return level_runtime._sha256(canonical)


def _build_configs(
    *,
    conditions: Sequence[DownstreamCondition],
    run_id: str,
    workflow: str,
    model: str | None,
    concurrency: int,
    config_overrides: Mapping[str, Mapping[str, Any]] | None,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    from dslighting.core.config.builder import ConfigBuilder

    builder = ConfigBuilder()
    configs: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for condition in conditions:
        condition_id = condition.condition_id
        try:
            build_kwargs = {
                key: dict(value) for key, value in (config_overrides or {}).items()
            }
            config = builder.build_config(
                workflow=workflow,
                model=model,
                task_context={
                    "policy": condition.task_context_policy,
                    "l1_artifact_dir": (
                        str(condition.annotations_dir)
                        if condition.annotations_dir is not None
                        else None
                    ),
                    "l2_guidance_path": None,
                    "require_canonical_layout": True,
                },
                run_name=f"{run_id}_{condition_id}",
                **build_kwargs,
            )
        except Exception as error:
            raise PreflightError(
                f"cannot build config for downstream condition {condition_id!r}: {error}"
            ) from error
        config.scheduler.max_concurrency = concurrency
        config.scheduler.llm_max_concurrency = concurrency
        config.scheduler.gpu_policy = "cpu_default"
        config.llm.max_concurrent_per_key = concurrency
        config.scheduler.run_id = run_id
        config.run.parameters = dict(config.run.parameters)
        config.run.parameters["output_artifact_suffix_seed"] = run_id
        if not dry_run and not config.llm.get_api_keys():
            raise PreflightError(
                "no LLM API key resolved; configure API_KEY/OPENAI_API_KEY or "
                "LLM_MODEL_CONFIGS before a non-dry run"
            )
        configs[condition_id] = config
        fingerprints[condition_id] = _fixed_config_fingerprint(config)

    if len(set(fingerprints.values())) != 1:
        raise PreflightError(
            "downstream conditions changed fixed runtime configuration: "
            + ", ".join(f"{key}={value}" for key, value in fingerprints.items())
        )
    return configs, next(iter(fingerprints.values()))


def _build_downstream_conditions(
    perception_skills: Sequence[PerceptionSkillCondition],
    *,
    include_main: bool,
) -> tuple[DownstreamCondition, ...]:
    conditions: list[DownstreamCondition] = []
    if include_main:
        conditions.append(
            DownstreamCondition(
                condition_id=MAIN_REFERENCE_ID,
                condition_type="main_reference",
                task_context_policy=MAIN_TASK_CONTEXT_POLICY,
            )
        )
    conditions.extend(
        DownstreamCondition(
            condition_id=skill.perception_skill_id,
            condition_type="perception_skill",
            task_context_policy=SKILL_TASK_CONTEXT_POLICY,
            perception_skill=skill,
        )
        for skill in perception_skills
    )
    if not conditions:
        raise PreflightError("downstream condition selection is empty")
    return tuple(conditions)


def _run_id(source_id: str) -> str:
    safe_source = "".join(character if character.isalnum() else "_" for character in source_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_source}_perception_skills_{timestamp}"


def _execute(args: argparse.Namespace) -> Path | None:
    catalog, descriptor, profile = level_runtime._resolve_benchmark_profile(
        str(args.benchmark).strip()
    )
    probe_root = level_runtime._resolve_directory(args.data_root, "benchmark data root")
    try:
        probe_registry = catalog.build_registry(descriptor, data_root=probe_root)
        available_tasks = probe_registry.list_competition_ids()
    except Exception as error:
        raise PreflightError(
            f"cannot initialize benchmark registry for {descriptor.source_id!r}: {error}"
        ) from error
    data_root = level_runtime._normalize_data_root(probe_root, available_tasks)
    registry = catalog.build_registry(descriptor, data_root=data_root)
    vendor_dir = level_runtime._resolve_directory(
        descriptor.registry_root, "benchmark registry"
    )
    targets = level_runtime._resolve_tasks(
        registry=registry,
        inline_tasks=args.tasks,
        tasks_file=args.tasks_file,
        limit=args.limit,
    )
    tasks = [target.task_id for target in targets]
    source_public_attestations = level_runtime._attest_source_public_trees(targets)

    perception_skills, skills_summary = _load_perception_skills(
        args.perception_skills_manifest,
        args.annotations_root,
        args.perception_skills,
    )
    conditions = _build_downstream_conditions(
        perception_skills,
        include_main=not args.exclude_main,
    )
    l1_summaries = {
        skill.perception_skill_id: level_runtime._preflight_l1(
            skill.annotations_dir, targets
        )
        for skill in perception_skills
    }

    workflow = str(args.workflow or profile.default_workflow)
    run_id = _run_id(descriptor.source_id)
    run_root = (RUNS_ROOT / run_id).resolve()
    configs, fixed_config_sha256 = _build_configs(
        conditions=conditions,
        run_id=run_id,
        workflow=workflow,
        model=args.model,
        concurrency=args.concurrency,
        config_overrides=profile.config_overrides,
        dry_run=args.dry_run,
    )
    first_config = configs[conditions[0].condition_id]

    source_manifest = None
    if descriptor.manifest_path is not None:
        payload = level_runtime._read_bytes(
            descriptor.manifest_path, "benchmark source manifest"
        )
        source_manifest = {
            "path": str(descriptor.manifest_path),
            "sha256": level_runtime._sha256(payload),
        }

    manifest_path = run_root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "perception_skills_downstream_ablation",
        "run_id": run_id,
        "status": "validated" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "created_at_utc": level_runtime._utc_now(),
        "benchmark": {
            "source_id": descriptor.source_id,
            "contract_id": descriptor.contract_id,
            "engine_id": descriptor.engine_id,
            "data_root": str(data_root),
            "registry_root": str(vendor_dir),
            "source_manifest": source_manifest,
        },
        "selection": {
            "task_count": len(tasks),
            "tasks": tasks,
            "dataset_ids": {target.task_id: target.dataset_id for target in targets},
            "public_dirs": {target.task_id: str(target.public_dir) for target in targets},
            "source_public_trees": source_public_attestations,
        },
        "runtime": {
            "task_context_policies": {
                condition.condition_id: condition.task_context_policy
                for condition in conditions
            },
            "workflow": workflow,
            "model": first_config.llm.model,
            "concurrency": args.concurrency,
            "llm_max_concurrency": first_config.scheduler.llm_max_concurrency,
            "llm_max_concurrent_per_key": first_config.llm.max_concurrent_per_key,
            "gpu_policy": first_config.scheduler.gpu_policy,
            "fixed_config_sha256": fixed_config_sha256,
            "output_contract": first_config.output_contract.model_dump(mode="json"),
        },
        "main_reference": {
            "included": not args.exclude_main,
            "condition_id": MAIN_REFERENCE_ID,
            "source": "benchmark_dataset_description",
        },
        "perception_skills": skills_summary,
        "l1_annotations": l1_summaries,
        "context_audit": {
            "fixed_fields": list(level_runtime.FIXED_CONTEXT_FIELDS),
            "status": "not_run" if args.dry_run else "pending",
        },
        "runs": [],
    }

    if args.dry_run:
        manifest["runs"] = [
            {
                "condition_id": condition.condition_id,
                "condition_type": condition.condition_type,
                "perception_skill_id": condition.perception_skill_id,
                "control": condition.is_control,
                "task_context_policy": condition.task_context_policy,
                "status": "validated",
            }
            for condition in conditions
        ]
        manifest["completed_at_utc"] = level_runtime._utc_now()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return None

    from dslighting.api.benchmark import DSBenchmark

    fixed_context_reference: dict[str, dict[str, str]] = {}
    completed_with_task_failures = False
    level_runtime._write_manifest(manifest_path, manifest)
    for condition in conditions:
        condition_id = condition.condition_id
        log_path = run_root / condition_id
        record: dict[str, Any] = {
            "condition_id": condition_id,
            "condition_type": condition.condition_type,
            "perception_skill_id": condition.perception_skill_id,
            "control": condition.is_control,
            "task_context_policy": condition.task_context_policy,
            "status": "running",
            "log_path": str(log_path),
        }
        manifest["runs"].append(record)
        level_runtime._write_manifest(manifest_path, manifest)
        try:
            benchmark = DSBenchmark(
                benchmark_type=descriptor.source_id,
                exp_name=f"{run_id}_{condition_id}",
                data_dir=str(data_root),
                vendor_comp_dir=str(vendor_dir),
                competitions=list(tasks),
            )
            result = benchmark.run(
                config=configs[condition_id],
                log_path=str(log_path),
                verbose=True,
            )
            audit_by_task = level_runtime._collect_task_context_audit(
                result,
                tasks=tasks,
                policy=condition.task_context_policy,
            )
            task_outcomes = level_runtime._collect_task_outcomes(
                result,
                audit_by_task,
                tasks=tasks,
                policy=condition_id,
            )
            level_runtime._verify_selected_artifacts(
                audit_by_task,
                policy=condition.task_context_policy,
                l1_summary=(
                    l1_summaries[condition.perception_skill_id]
                    if condition.perception_skill_id is not None
                    else None
                ),
                l2_summary=None,
            )
            level_runtime._verify_fixed_context(
                audit_by_task,
                fixed_context_reference,
                policy=condition_id,
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_at_utc": level_runtime._utc_now(),
                }
            )
            manifest["status"] = "failed"
            manifest["context_audit"] = {
                "fixed_fields": list(level_runtime.FIXED_CONTEXT_FIELDS),
                "status": "failed",
                "error": str(error),
                "reference": fixed_context_reference,
            }
            manifest["completed_at_utc"] = level_runtime._utc_now()
            level_runtime._write_manifest(manifest_path, manifest)
            raise

        condition_has_task_failures = any(
            outcome["status"] != "completed" for outcome in task_outcomes.values()
        )
        completed_with_task_failures = (
            completed_with_task_failures or condition_has_task_failures
        )
        record.update(
            {
                "status": (
                    "completed_with_task_failures"
                    if condition_has_task_failures
                    else "completed"
                ),
                "result_count": level_runtime._result_count(result),
                "results_path": level_runtime._result_path(result, "results_path"),
                "metadata_path": level_runtime._result_path(result, "metadata_path"),
                "task_context_audit": audit_by_task,
                "task_outcomes": task_outcomes,
                "completed_at_utc": level_runtime._utc_now(),
            }
        )
        manifest["context_audit"] = {
            "fixed_fields": list(level_runtime.FIXED_CONTEXT_FIELDS),
            "status": "verified_so_far",
            "reference": fixed_context_reference,
        }
        level_runtime._write_manifest(manifest_path, manifest)

    manifest["status"] = (
        "completed_with_task_failures" if completed_with_task_failures else "completed"
    )
    manifest["context_audit"] = {
        "fixed_fields": list(level_runtime.FIXED_CONTEXT_FIELDS),
        "status": "verified",
        "reference": fixed_context_reference,
    }
    manifest["completed_at_utc"] = level_runtime._utc_now()
    level_runtime._write_manifest(manifest_path, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest_path = _execute(args)
    except PreflightError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2
    except ExperimentInvariantError as error:
        print(f"experiment invariant failed: {error}", file=sys.stderr)
        return 3

    if manifest_path is not None:
        print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
