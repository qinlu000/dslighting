#!/usr/bin/env python3
"""Run main/L1/L2/L3 task-context ablations on reviewed benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = "dabench"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
SUPPORTED_POLICIES = ("main", "l1", "l2", "l3")
FIXED_CONTEXT_FIELDS = (
    "main_data_report_sha256",
    "io_instructions_sha256",
    "output_artifact_name",
    "submission_contract_sha256",
    "evaluation_contract_ref_sha256",
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PreflightError(RuntimeError):
    """Raised before any benchmark starts when an ablation input is invalid."""


class ExperimentInvariantError(RuntimeError):
    """Raised when completed arms differ outside the declared treatment."""


@dataclass(frozen=True)
class TaskTarget:
    """Prepared, solver-visible input paths resolved by the source registry."""

    task_id: str
    dataset_id: str
    public_dir: Path
    sample_submission_path: Path | None
    description_text: str
    description_sha256: str


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _policy_list(raw: str) -> tuple[str, ...]:
    policies = tuple(token.strip().lower() for token in raw.split(",") if token.strip())
    if not policies:
        raise argparse.ArgumentTypeError("expected at least one policy")
    invalid = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unsupported policies: " + ", ".join(invalid) + "; expected only main,l1,l2,l3"
        )
    duplicates = sorted({policy for policy in policies if policies.count(policy) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError("duplicate policies: " + ", ".join(duplicates))
    return policies


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark task-context ablation: main, L1, L2, and L3."
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Reviewed benchmark source ID (default: {DEFAULT_BENCHMARK}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Benchmark data root containing <task_id>/prepared/public.",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        nargs="+",
        help="Task IDs separated by spaces or commas.",
    )
    task_group.add_argument(
        "--tasks-file",
        type=Path,
        help="UTF-8 file with one task ID per line; blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Use only the first N resolved tasks.",
    )
    parser.add_argument(
        "--policies",
        type=_policy_list,
        default=SUPPORTED_POLICIES,
        help="Comma-separated policy subset (default: main,l1,l2,l3).",
    )
    parser.add_argument(
        "--workflow",
        help="Override the benchmark profile's default DSLighting workflow.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name passed to ConfigBuilder.",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="Maximum concurrent benchmark tasks (default: 1).",
    )
    parser.add_argument(
        "--l1-artifact-dir",
        type=Path,
        help=(
            "Directory containing one <resolved_dataset_id>.json L1 artifact "
            "per selected dataset; required by l1/l3."
        ),
    )
    parser.add_argument(
        "--l2-guidance-path",
        type=Path,
        help="L2 Markdown artifact (default: the reviewed benchmark-specific artifact).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and configs and print a JSON plan without writing or running tasks.",
    )
    return parser


def _resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a readable directory: {resolved}")
    return resolved


def _split_task_tokens(values: Sequence[str]) -> list[str]:
    tasks: list[str] = []
    for value in values:
        tasks.extend(token.strip() for token in value.split(",") if token.strip())
    return tasks


def _read_tasks_file(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PreflightError(f"cannot read tasks file {resolved}: {error}") from error

    tokens: list[str] = []
    for line in text.splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            tokens.extend(token.strip() for token in content.split(",") if token.strip())
    return tokens


def _validate_task_ids(tasks: Sequence[str], *, available_tasks: set[str]) -> list[str]:
    if not tasks:
        raise PreflightError("task selection is empty")

    duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
    if duplicates:
        raise PreflightError(f"duplicate task IDs: {', '.join(duplicates)}")

    invalid = [
        task
        for task in tasks
        if not task.strip()
        or Path(task).name != task
        or task in {".", ".."}
        or "/" in task
        or "\\" in task
    ]
    if invalid:
        raise PreflightError(
            "task IDs must be safe registry basenames; invalid IDs: " + ", ".join(invalid)
        )
    unknown = sorted(set(tasks) - available_tasks)
    if unknown:
        raise PreflightError(
            "selected tasks are not in the benchmark registry: " + ", ".join(unknown)
        )
    return list(tasks)


def _normalize_data_root(path: Path, available_tasks: Sequence[str]) -> Path:
    """Accept either a task root collection or its conventional parent."""

    root = _resolve_directory(path, "benchmark data root")
    candidates = [root]
    nested = root / "competitions"
    if nested.is_dir():
        candidates.append(nested.resolve())

    for candidate in candidates:
        if any((candidate / task / "prepared" / "public").is_dir() for task in available_tasks):
            return candidate
    return root


def _resolve_tasks(
    *,
    registry: Any,
    inline_tasks: Sequence[str] | None,
    tasks_file: Path | None,
    limit: int | None,
) -> list[TaskTarget]:
    from dslighting.benchmark.vendor.mlebench.data import is_dataset_prepared
    from dslighting.core.tasks.resolver import TaskResolver

    available = tuple(registry.list_competition_ids())
    available_set = set(available)
    if inline_tasks:
        tasks = _split_task_tokens(inline_tasks)
    elif tasks_file is not None:
        tasks = _read_tasks_file(tasks_file)
    else:
        tasks = sorted(available)

    tasks = _validate_task_ids(tasks, available_tasks=available_set)
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise PreflightError("task selection is empty after applying --limit")

    targets: list[TaskTarget] = []
    for task in tasks:
        try:
            competition = registry.get_competition(task)
        except Exception as error:
            raise PreflightError(
                f"cannot resolve registry contract for {task!r}: {error}"
            ) from error
        if not is_dataset_prepared(competition, grading_only=False):
            raise PreflightError(
                f"selected task {task!r} is not fully prepared under the benchmark data root"
            )
        public_dir = Path(competition.public_dir).resolve()
        if not public_dir.is_dir():
            raise PreflightError(
                f"selected task {task!r} has no prepared public directory: {public_dir}"
            )
        sample = getattr(competition, "sample_submission", None)
        try:
            task_config = registry.resolve_task_config(task).config
            dataset_id = TaskResolver.resolve_dataset_id(
                task_id=task,
                config=task_config,
                context_dataset_id_field=(registry.descriptor.context_dataset_id_field),
                context_dataset_id_prefix=(registry.descriptor.context_dataset_id_prefix),
            )
        except Exception as error:
            raise PreflightError(
                f"cannot resolve semantic dataset identity for {task!r}: {error}"
            ) from error
        targets.append(
            TaskTarget(
                task_id=task,
                dataset_id=dataset_id,
                public_dir=public_dir,
                sample_submission_path=(
                    Path(sample).resolve() if isinstance(sample, Path) else None
                ),
                description_text=str(competition.description),
                description_sha256=_sha256(str(competition.description).encode("utf-8")),
            )
        )
    return targets


def _attest_source_public_trees(targets: Sequence[TaskTarget]) -> dict[str, dict[str, Any]]:
    from dslighting.core.task_context import attest_public_tree

    attestations: dict[str, dict[str, Any]] = {}
    for target in targets:
        try:
            attestation = attest_public_tree(target.public_dir)
        except Exception as error:
            raise PreflightError(
                f"cannot attest public data for {target.task_id!r}: {error}"
            ) from error
        attestations[target.task_id] = attestation.as_dict()
    return attestations


def _resolve_benchmark_profile(source_id: str) -> tuple[Any, Any, Any]:
    from dslighting.benchmark.core.source_catalog import get_benchmark_source_catalog
    from experiments.data_card_ablation.benchmark_profiles import get_ablation_profile

    catalog = get_benchmark_source_catalog()
    try:
        descriptor = catalog.get_source(source_id)
    except Exception as error:
        raise PreflightError(f"unknown benchmark source {source_id!r}: {error}") from error
    if descriptor.contract_id != "mle_task_contract/v1" or descriptor.engine_id != "mle":
        raise PreflightError(
            f"benchmark source {source_id!r} uses unsupported runtime capability "
            f"contract={descriptor.contract_id!r}, engine={descriptor.engine_id!r}"
        )
    try:
        profile = get_ablation_profile(descriptor.source_id)
    except ValueError as error:
        raise PreflightError(str(error)) from error
    return catalog, descriptor, profile


def _read_bytes(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise PreflightError(f"missing {label}: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PreflightError(f"cannot read {label} {path}: {error}") from error
    if not payload:
        raise PreflightError(f"{label} is empty: {path}")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _preflight_l1(
    l1_artifact_dir: Path,
    targets: Sequence[TaskTarget],
) -> dict[str, Any]:
    from dslighting.core.task_context import (
        load_l1_artifact,
        validate_l1_public_coverage,
    )

    artifact_dir = _resolve_directory(l1_artifact_dir, "L1 artifact directory")
    bundle = hashlib.sha256()
    files: dict[str, dict[str, Any]] = {}
    artifact_digests: dict[str, tuple[Path, str]] = {}

    for target in targets:
        task = target.task_id
        artifact_id = target.dataset_id
        path = artifact_dir / f"{artifact_id}.json"
        if artifact_id not in artifact_digests:
            payload = _read_bytes(path, f"L1 artifact for dataset {artifact_id}")
            file_digest = _sha256(payload)
            artifact_digests[artifact_id] = (path.resolve(), file_digest)
            bundle.update(artifact_id.encode("utf-8"))
            bundle.update(b"\0")
            bundle.update(file_digest.encode("ascii"))
            bundle.update(b"\n")
        try:
            artifact = load_l1_artifact(
                artifact_dir,
                artifact_id,
                expected_dataset_id=artifact_id,
                expected_task_id=task,
            )
            excluded_paths: list[Path] = []
            if target.sample_submission_path is not None:
                excluded_paths.append(target.sample_submission_path)
            public_attestation = validate_l1_public_coverage(
                artifact,
                target.public_dir,
                excluded_paths=excluded_paths,
            )
            rendered_l1 = artifact.render_markdown(heading_level=3)
            treated_description = artifact.replace_dataset_description(
                target.description_text
            )
        except Exception as error:
            raise PreflightError(f"invalid L1 artifact {path}: {error}") from error

        resolved_path, file_digest = artifact_digests[artifact_id]
        files[task] = {
            "artifact_id": artifact_id,
            "path": str(resolved_path),
            "sha256": file_digest,
            "source_description_sha256": target.description_sha256,
            "rendered_l1_sha256": _sha256(rendered_l1.encode("utf-8")),
            "rendered_l1_chars": len(rendered_l1),
            "treated_description_sha256": _sha256(
                treated_description.encode("utf-8")
            ),
            "semantic_public_tree": public_attestation.as_dict(),
        }

    return {
        "directory": str(artifact_dir),
        "validated_task_count": len(targets),
        "artifact_count": len(artifact_digests),
        "bundle_sha256": bundle.hexdigest(),
        "files": files,
    }


def _preflight_l2(l2_guidance_path: Path) -> tuple[Path, dict[str, Any]]:
    from dslighting.core.task_context import load_l2_artifact

    path = l2_guidance_path.expanduser().resolve()
    if path.suffix.lower() not in {".md", ".markdown"}:
        raise PreflightError(f"L2 guidance must be a Markdown file: {path}")
    payload = _read_bytes(path, "L2 guidance")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise PreflightError(f"L2 guidance is not valid UTF-8: {path}") from error
    if not text.strip():
        raise PreflightError(f"L2 guidance contains no text: {path}")
    try:
        load_l2_artifact(path)
    except Exception as error:
        raise PreflightError(f"invalid L2 guidance {path}: {error}") from error
    return path, {"path": str(path), "sha256": _sha256(payload)}


def _build_configs(
    *,
    policies: Sequence[str],
    run_id: str,
    workflow: str,
    model: str,
    concurrency: int,
    l1_artifact_dir: Path | None,
    l2_guidance_path: Path | None,
    config_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    dry_run: bool,
) -> dict[str, Any]:
    from dslighting.core.config.builder import ConfigBuilder

    builder = ConfigBuilder()
    configs: dict[str, Any] = {}
    for policy in policies:
        try:
            build_kwargs: dict[str, Any] = {
                key: dict(value) for key, value in (config_overrides or {}).items()
            }
            config = builder.build_config(
                workflow=workflow,
                model=model,
                task_context={
                    "policy": policy,
                    "l1_artifact_dir": (
                        str(l1_artifact_dir) if l1_artifact_dir is not None else None
                    ),
                    "l2_guidance_path": (
                        str(l2_guidance_path) if l2_guidance_path is not None else None
                    ),
                    "require_canonical_layout": True,
                },
                run_name=f"{run_id}_{policy}",
                **build_kwargs,
            )
        except Exception as error:
            raise PreflightError(f"cannot build config for policy {policy!r}: {error}") from error
        config.scheduler.max_concurrency = concurrency
        # Scheduler identity remains available to the benchmark runtime.  Output
        # naming is a separate, explicit opt-in so ordinary benchmark runs that
        # happen to set a run ID retain main's random suffix behavior.
        config.scheduler.run_id = run_id
        config.run.parameters = dict(config.run.parameters)
        config.run.parameters["output_artifact_suffix_seed"] = run_id
        if not dry_run and not config.llm.get_api_keys():
            raise PreflightError(
                "no LLM API key resolved; configure OPENAI_API_KEY or LLM_MODEL_CONFIGS "
                "before a non-dry run"
            )
        configs[policy] = config
    return configs


def _fixed_config_fingerprint(config: Any) -> str:
    """Fingerprint the complete config after removing declared arm variables."""

    payload = config.model_dump(mode="json")
    run = payload.get("run")
    if isinstance(run, dict):
        run.pop("run_name", None)
    task_context = payload.get("task_context")
    if isinstance(task_context, dict):
        for field in ("policy", "l1_artifact_dir", "l2_guidance_path"):
            task_context.pop(field, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(canonical)


def _verify_fixed_configs(configs: Mapping[str, Any]) -> str:
    fingerprints = {policy: _fixed_config_fingerprint(config) for policy, config in configs.items()}
    unique = set(fingerprints.values())
    if len(unique) != 1:
        raise PreflightError(
            "ablation policies changed main-owned runtime configuration: "
            + ", ".join(f"{policy}={digest}" for policy, digest in fingerprints.items())
        )
    return next(iter(unique))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(source_id: str = DEFAULT_BENCHMARK) -> str:
    safe_source = "".join(character if character.isalnum() else "_" for character in source_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_source}_data_card_{timestamp}"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result_count(result: Any) -> int | None:
    results = getattr(result, "results", None)
    if results is None:
        return None
    try:
        return len(results)
    except TypeError:
        return None


def _result_path(result: Any, field: str) -> str | None:
    value = getattr(result, field, None)
    return str(value) if value is not None else None


def _collect_task_context_audit(
    benchmark: Any,
    *,
    tasks: Sequence[str],
    policy: str,
) -> dict[str, dict[str, Any]]:
    runner = getattr(benchmark, "runner", None)
    get_records = getattr(runner, "get_run_records", None)
    if not callable(get_records):
        raise ExperimentInvariantError(f"policy {policy!r} did not expose task run records")

    expected_tasks = set(tasks)
    audit_by_task: dict[str, dict[str, Any]] = {}
    run_summary_by_task: dict[str, dict[str, Any]] = {}
    attempt_count_by_task: dict[str, int] = {}
    for record in get_records():
        if not isinstance(record, Mapping):
            continue
        task_id = str(record.get("task_id") or "")
        if task_id not in expected_tasks:
            continue
        summary = record.get("summary")
        if not isinstance(summary, Mapping) or not isinstance(summary.get("success"), bool):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} did not persist a boolean run outcome"
            )
        attempt_count_by_task[task_id] = attempt_count_by_task.get(task_id, 0) + 1
        run_summary_by_task[task_id] = {
            "success": summary["success"],
            "result": summary.get("result"),
        }
        runtime_audit = record.get("task_context_audit")
        if not isinstance(runtime_audit, Mapping):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} did not persist task-context audit"
            )
        provenance = runtime_audit.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} did not persist task-context provenance"
            )
        if provenance.get("policy") != policy:
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} recorded policy "
                f"{provenance.get('policy')!r}"
            )
        if (
            not isinstance(provenance.get("dataset_id"), str)
            or not str(provenance.get("dataset_id")).strip()
        ):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} recorded no semantic dataset identity"
            )
        expects_l1 = policy in {"l1", "l3"}
        expects_l2 = policy in {"l2", "l3"}
        if (provenance.get("l1") is not None) != expects_l1 or (
            provenance.get("l2") is not None
        ) != expects_l2:
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} violates the L1/L2 treatment matrix"
            )
        fixed_context = runtime_audit.get("fixed_context")
        if not isinstance(fixed_context, Mapping):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} did not persist fixed task context"
            )
        context_record = {
            "task_context_provenance": dict(provenance),
            "fixed_task_context": dict(fixed_context),
        }
        previous = audit_by_task.get(task_id)
        if previous is not None:
            if previous == context_record:
                continue
            raise ExperimentInvariantError(
                f"policy {policy!r} produced conflicting retry records for task {task_id!r}"
            )
        audit_by_task[task_id] = context_record

    missing = sorted(expected_tasks - audit_by_task.keys())
    if missing:
        raise ExperimentInvariantError(
            f"policy {policy!r} has no auditable context for tasks: {', '.join(missing)}"
        )
    return {
        task: {
            **audit_by_task[task],
            "run_summary": {
                **run_summary_by_task[task],
                "attempt_count": attempt_count_by_task[task],
            },
        }
        for task in tasks
    }


def _collect_task_outcomes(
    benchmark: Any,
    audit_by_task: Mapping[str, Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    policy: str,
) -> dict[str, dict[str, Any]]:
    get_columns = getattr(benchmark, "get_result_columns", None)
    rows = getattr(benchmark, "results", None)
    if not callable(get_columns) or not isinstance(rows, list):
        raise ExperimentInvariantError(f"policy {policy!r} did not expose benchmark task outcomes")

    columns = get_columns()
    required = {
        "competition_id",
        "score",
        "submission_exists",
        "valid_submission",
        "error_message",
    }
    if not isinstance(columns, list) or not required.issubset(columns):
        raise ExperimentInvariantError(
            f"policy {policy!r} exposed an incompatible benchmark result schema"
        )

    expected_tasks = set(tasks)
    outcomes: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != len(columns):
            raise ExperimentInvariantError(
                f"policy {policy!r} exposed a malformed benchmark result row"
            )
        row = dict(zip(columns, raw_row))
        task_id = str(row.get("competition_id") or "")
        if task_id not in expected_tasks:
            continue
        if task_id in outcomes:
            raise ExperimentInvariantError(
                f"policy {policy!r} produced duplicate benchmark outcomes for {task_id!r}"
            )
        submission_exists = row.get("submission_exists")
        valid_submission = row.get("valid_submission")
        if not isinstance(submission_exists, bool) or not isinstance(valid_submission, bool):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} has non-boolean submission status"
            )
        run_summary = audit_by_task[task_id].get("run_summary")
        if not isinstance(run_summary, Mapping) or not isinstance(run_summary.get("success"), bool):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} has no auditable workflow outcome"
            )
        workflow_success = run_summary["success"]
        status = (
            "workflow_failed"
            if not workflow_success
            else "completed" if valid_submission else "invalid_submission"
        )
        outcomes[task_id] = {
            "status": status,
            "workflow_success": workflow_success,
            "submission_exists": submission_exists,
            "valid_submission": valid_submission,
            "score": row.get("score"),
            "error_message": row.get("error_message"),
        }

    missing = sorted(expected_tasks - outcomes.keys())
    if missing:
        raise ExperimentInvariantError(
            f"policy {policy!r} has no benchmark outcome for tasks: {', '.join(missing)}"
        )
    return {task: outcomes[task] for task in tasks}


def _verify_fixed_context(
    audit_by_task: Mapping[str, Mapping[str, Any]],
    reference: dict[str, dict[str, str]],
    *,
    policy: str,
) -> None:
    for task_id, context_record in audit_by_task.items():
        provenance = context_record.get("task_context_provenance")
        fixed_context = context_record.get("fixed_task_context")
        if not isinstance(provenance, Mapping) or not isinstance(fixed_context, Mapping):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} has malformed audit data"
            )
        fingerprint: dict[str, str] = {}
        values = {
            "main_data_report_sha256": provenance.get("main_data_report_sha256"),
            "io_instructions_sha256": fixed_context.get("io_instructions_sha256"),
            "output_artifact_name": fixed_context.get("output_artifact_name"),
            "submission_contract_sha256": fixed_context.get(
                "submission_contract_sha256"
            ),
            "evaluation_contract_ref_sha256": fixed_context.get(
                "evaluation_contract_ref_sha256"
            ),
        }
        for field in FIXED_CONTEXT_FIELDS:
            value = values[field]
            if not isinstance(value, str) or not value:
                raise ExperimentInvariantError(f"policy {policy!r} task {task_id!r} has no {field}")
            fingerprint[field] = value

        expected = reference.setdefault(task_id, fingerprint)
        if fingerprint != expected:
            changed = [
                field for field in FIXED_CONTEXT_FIELDS if fingerprint[field] != expected[field]
            ]
            raise ExperimentInvariantError(
                f"policy {policy!r} changed fixed task context for {task_id!r}: "
                + ", ".join(changed)
            )


def _verify_selected_artifacts(
    audit_by_task: Mapping[str, Mapping[str, Any]],
    *,
    policy: str,
    l1_summary: Mapping[str, Any] | None,
    l2_summary: Mapping[str, Any] | None,
) -> None:
    for task_id, context_record in audit_by_task.items():
        provenance = context_record.get("task_context_provenance")
        if not isinstance(provenance, Mapping):
            raise ExperimentInvariantError(
                f"policy {policy!r} task {task_id!r} has malformed provenance"
            )

        if policy in {"l1", "l3"}:
            files = l1_summary.get("files") if isinstance(l1_summary, Mapping) else None
            expected = files.get(task_id) if isinstance(files, Mapping) else None
            expected_provenance = (
                {"path": expected.get("path"), "sha256": expected.get("sha256")}
                if isinstance(expected, Mapping)
                else None
            )
            expected_artifact_id = (
                expected.get("artifact_id") if isinstance(expected, Mapping) else None
            )
            if (
                expected_provenance is None
                or provenance.get("l1") != expected_provenance
                or provenance.get("dataset_id") != expected_artifact_id
            ):
                raise ExperimentInvariantError(
                    f"policy {policy!r} task {task_id!r} did not use the preflighted L1 artifact"
                )

        if policy in {"l2", "l3"}:
            expected = (
                {"path": l2_summary.get("path"), "sha256": l2_summary.get("sha256")}
                if isinstance(l2_summary, Mapping)
                else None
            )
            if expected is None or provenance.get("l2") != expected:
                raise ExperimentInvariantError(
                    f"policy {policy!r} task {task_id!r} did not use the preflighted L2 artifact"
                )


def _execute(args: argparse.Namespace) -> Path | None:
    catalog, descriptor, profile = _resolve_benchmark_profile(str(args.benchmark).strip())
    probe_root = _resolve_directory(args.data_root, "benchmark data root")
    try:
        probe_registry = catalog.build_registry(descriptor, data_root=probe_root)
        available_tasks = probe_registry.list_competition_ids()
    except Exception as error:
        raise PreflightError(
            f"cannot initialize benchmark registry for {descriptor.source_id!r}: {error}"
        ) from error
    data_root = _normalize_data_root(probe_root, available_tasks)
    registry = catalog.build_registry(descriptor, data_root=data_root)
    vendor_dir = _resolve_directory(descriptor.registry_root, "benchmark registry")

    targets = _resolve_tasks(
        registry=registry,
        inline_tasks=args.tasks,
        tasks_file=args.tasks_file,
        limit=args.limit,
    )
    tasks = [target.task_id for target in targets]
    policies = tuple(args.policies)
    source_public_attestations = _attest_source_public_trees(targets)

    l1_summary = None
    l1_artifact_dir = None
    if any(policy in {"l1", "l3"} for policy in policies):
        if args.l1_artifact_dir is None:
            raise PreflightError("--l1-artifact-dir is required by selected policy l1/l3")
        l1_summary = _preflight_l1(args.l1_artifact_dir, targets)
        l1_artifact_dir = Path(l1_summary["directory"])

    l2_summary = None
    l2_guidance_path = None
    if any(policy in {"l2", "l3"} for policy in policies):
        requested_l2 = args.l2_guidance_path or profile.l2_guidance_path
        l2_guidance_path, l2_summary = _preflight_l2(requested_l2)

    workflow = str(args.workflow or profile.default_workflow)
    run_id = _run_id(descriptor.source_id)
    run_root = (RUNS_ROOT / run_id).resolve()
    configs = _build_configs(
        policies=policies,
        run_id=run_id,
        workflow=workflow,
        model=args.model,
        concurrency=args.concurrency,
        l1_artifact_dir=l1_artifact_dir,
        l2_guidance_path=l2_guidance_path,
        config_overrides=profile.config_overrides,
        dry_run=args.dry_run,
    )
    fixed_config_sha256 = _verify_fixed_configs(configs)

    source_manifest = None
    if descriptor.manifest_path is not None:
        payload = _read_bytes(descriptor.manifest_path, "benchmark source manifest")
        source_manifest = {
            "path": str(descriptor.manifest_path),
            "sha256": _sha256(payload),
        }

    manifest_path = run_root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "experiment": "data_card_ablation",
        "run_id": run_id,
        "status": "validated" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "created_at_utc": _utc_now(),
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
            "descriptions": {
                target.task_id: {
                    "sha256": target.description_sha256,
                    "chars": len(target.description_text),
                }
                for target in targets
            },
            "source_public_trees": source_public_attestations,
        },
        "runtime": {
            "workflow": workflow,
            "model": args.model,
            "concurrency": args.concurrency,
            "fixed_config_sha256": fixed_config_sha256,
            "output_contract": configs[policies[0]].output_contract.model_dump(mode="json"),
        },
        "policies": list(policies),
        "artifacts": {
            key: value
            for key, value in (("l1", l1_summary), ("l2", l2_summary))
            if value is not None
        },
        "context_audit": {
            "fixed_fields": list(FIXED_CONTEXT_FIELDS),
            "status": "not_run" if args.dry_run else "pending",
        },
        "runs": [],
    }

    if args.dry_run:
        manifest["runs"] = [{"policy": policy, "status": "validated"} for policy in policies]
        manifest["completed_at_utc"] = _utc_now()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return None

    from dslighting.api.benchmark import DSBenchmark

    fixed_context_reference: dict[str, dict[str, str]] = {}
    completed_with_task_failures = False
    _write_manifest(manifest_path, manifest)
    for policy in policies:
        log_path = run_root / policy
        record: dict[str, Any] = {
            "policy": policy,
            "status": "running",
            "log_path": str(log_path),
        }
        manifest["runs"].append(record)
        _write_manifest(manifest_path, manifest)

        try:
            benchmark = DSBenchmark(
                benchmark_type=descriptor.source_id,
                exp_name=f"{run_id}_{policy}",
                data_dir=str(data_root),
                vendor_comp_dir=str(vendor_dir),
                competitions=list(tasks),
            )
            result = benchmark.run(
                config=configs[policy],
                log_path=str(log_path),
                verbose=True,
            )
            audit_by_task = _collect_task_context_audit(
                result,
                tasks=tasks,
                policy=policy,
            )
            task_outcomes = _collect_task_outcomes(
                result,
                audit_by_task,
                tasks=tasks,
                policy=policy,
            )
            _verify_selected_artifacts(
                audit_by_task,
                policy=policy,
                l1_summary=l1_summary,
                l2_summary=l2_summary,
            )
            _verify_fixed_context(
                audit_by_task,
                fixed_context_reference,
                policy=policy,
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_at_utc": _utc_now(),
                }
            )
            manifest["status"] = "failed"
            manifest["context_audit"] = {
                "fixed_fields": list(FIXED_CONTEXT_FIELDS),
                "status": "failed",
                "error": str(error),
                "reference": fixed_context_reference,
            }
            manifest["completed_at_utc"] = _utc_now()
            _write_manifest(manifest_path, manifest)
            raise error

        arm_has_task_failures = any(
            outcome["status"] != "completed" for outcome in task_outcomes.values()
        )
        completed_with_task_failures = completed_with_task_failures or arm_has_task_failures
        record.update(
            {
                "status": (
                    "completed_with_task_failures" if arm_has_task_failures else "completed"
                ),
                "result_count": _result_count(result),
                "results_path": _result_path(result, "results_path"),
                "metadata_path": _result_path(result, "metadata_path"),
                "task_context_audit": audit_by_task,
                "task_outcomes": task_outcomes,
                "completed_at_utc": _utc_now(),
            }
        )
        manifest["context_audit"] = {
            "fixed_fields": list(FIXED_CONTEXT_FIELDS),
            "status": "verified_so_far",
            "reference": fixed_context_reference,
        }
        _write_manifest(manifest_path, manifest)

    manifest["status"] = (
        "completed_with_task_failures" if completed_with_task_failures else "completed"
    )
    manifest["context_audit"] = {
        "fixed_fields": list(FIXED_CONTEXT_FIELDS),
        "status": "verified",
        "reference": fixed_context_reference,
    }
    manifest["completed_at_utc"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
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
