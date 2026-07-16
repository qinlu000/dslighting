"""Shared condition execution engine for Data Card ablation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FIXED_CONTEXT_FIELDS = (
    "main_data_report_sha256",
    "io_instructions_sha256",
    "output_artifact_name",
    "submission_contract_sha256",
    "evaluation_contract_ref_sha256",
)
COMPLETED_STATUSES = frozenset({"completed", "completed_with_task_failures"})


class PreflightError(RuntimeError):
    """Raised before a paid benchmark starts when an input is invalid."""


class ExperimentInvariantError(RuntimeError):
    """Raised when conditions differ outside their declared treatment."""


class ManifestStoreError(RuntimeError):
    """Raised when experiment state cannot be read or committed safely."""


@dataclass(frozen=True)
class TaskTarget:
    """Prepared, solver-visible input resolved by the benchmark registry."""

    task_id: str
    dataset_id: str
    public_dir: Path
    sample_submission_path: Path | None
    description_text: str
    description_sha256: str


@dataclass(frozen=True)
class ExperimentCondition:
    """The complete declared treatment for one downstream condition."""

    condition_id: str
    task_context_policy: str
    l1_artifact_dir: Path | None = None
    l2_guidance_path: Path | None = None
    condition_type: str = "data_card_level"
    treatment_id: str | None = None
    control: bool = False

    def as_manifest(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type,
            "treatment_id": self.treatment_id,
            "control": self.control,
            "task_context_policy": self.task_context_policy,
        }


@dataclass(frozen=True)
class ConditionRuntime:
    """Runtime values shared by every condition in one protocol."""

    workflow: str
    model: str
    task_concurrency: int
    llm_global_concurrency: int | None = None
    llm_per_key_concurrency: int | None = None
    llm_max_retries: int | None = None
    llm_thinking: bool | None = None
    sandbox_timeout_seconds: int | None = None
    sandbox_backend: str | None = None
    sandbox_environment_policy: str | None = None
    sandbox_network_policy: str | None = None
    device_admission: str | None = None
    checkpoint_resume_enabled: bool | None = None


@dataclass(frozen=True)
class PreparedBenchmark:
    """Benchmark-native paths and immutable selected task inputs."""

    descriptor: Any
    data_root: Path
    vendor_dir: Path
    targets: tuple[TaskTarget, ...]
    source_public_attestations: Mapping[str, Mapping[str, Any]]

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(target.task_id for target in self.targets)


@dataclass(frozen=True)
class EngineHooks:
    """Optional benchmark-specific attestations around each condition."""

    scoring_inputs: Callable[[str, str], Mapping[str, Any] | None] | None = None
    source_guard: Callable[[], None] | None = None


@dataclass(frozen=True)
class EngineRun:
    """One repetition's immutable configs and condition inputs."""

    repetition: int
    run_id: str
    configs: Mapping[str, Any]
    l1_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    l2_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class ManifestStore:
    """Atomic state storage for one experiment directory."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.path = self.run_root / "manifest.json"

    def write(self, manifest: Mapping[str, Any]) -> None:
        self.write_path(self.path, manifest)

    @staticmethod
    def write_path(path: Path, manifest: Mapping[str, Any]) -> None:
        resolved = path.expanduser().resolve()
        if resolved.is_symlink():
            raise ManifestStoreError(f"manifest path must not be a symlink: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
        payload = (json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, resolved)
            parent_descriptor = os.open(resolved.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError as error:
            raise ManifestStoreError(f"cannot commit manifest {resolved}: {error}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.is_file() or self.path.is_symlink():
            raise ManifestStoreError(f"manifest is missing or unsafe: {self.path}")
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                value = json.load(stream, object_pairs_hook=self._reject_duplicate_keys)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ManifestStoreError(f"invalid manifest {self.path}: {error}") from error
        if not isinstance(value, dict):
            raise ManifestStoreError(f"manifest must be a JSON object: {self.path}")
        return value

    def archive_log(self, log_path: Path, record_id: str) -> str | None:
        if not log_path.exists():
            return None
        if log_path.is_symlink():
            raise ManifestStoreError(f"condition log path must not be a symlink: {log_path}")
        archive_root = self.run_root / "resume_attempts" / record_id
        if archive_root.is_symlink():
            raise ManifestStoreError(f"condition archive must not be a symlink: {archive_root}")
        destination = archive_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            archive_root.mkdir(parents=True, exist_ok=True)
            os.replace(log_path, destination)
        except OSError as error:
            raise ManifestStoreError(f"cannot archive prior log {log_path}: {error}") from error
        return str(destination.resolve())

    @staticmethod
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_bytes(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise PreflightError(f"missing {label}: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PreflightError(f"cannot read {label} {path}: {error}") from error
    if not payload:
        raise PreflightError(f"{label} is empty: {path}")
    return payload


def resolve_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a readable directory: {resolved}")
    return resolved


def split_task_tokens(values: Sequence[str]) -> list[str]:
    tasks: list[str] = []
    for value in values:
        tasks.extend(token.strip() for token in value.split(",") if token.strip())
    return tasks


def read_tasks_file(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PreflightError(f"cannot read tasks file {resolved}: {error}") from error
    tasks: list[str] = []
    for line in text.splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            tasks.extend(token.strip() for token in content.split(",") if token.strip())
    return tasks


def validate_task_ids(tasks: Sequence[str], *, available_tasks: set[str]) -> list[str]:
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


def normalize_data_root(path: Path, available_tasks: Sequence[str]) -> Path:
    root = resolve_directory(path, "benchmark data root")
    candidates = [root]
    nested = root / "competitions"
    if nested.is_dir():
        candidates.append(nested.resolve())
    for candidate in candidates:
        if any((candidate / task / "prepared" / "public").is_dir() for task in available_tasks):
            return candidate
    return root


def resolve_tasks(
    *,
    registry: Any,
    inline_tasks: Sequence[str] | None,
    tasks_file: Path | None,
    limit: int | None,
) -> list[TaskTarget]:
    from dslighting.benchmark.vendor.mlebench.data import is_dataset_prepared
    from dslighting.core.tasks.resolver import TaskResolver

    available = tuple(registry.list_competition_ids())
    if inline_tasks:
        tasks = split_task_tokens(inline_tasks)
    elif tasks_file is not None:
        tasks = read_tasks_file(tasks_file)
    else:
        tasks = sorted(available)
    tasks = validate_task_ids(tasks, available_tasks=set(available))
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise PreflightError("task selection is empty after applying --limit")

    targets: list[TaskTarget] = []
    for task in tasks:
        try:
            competition = registry.get_competition(task)
        except Exception as error:
            raise PreflightError(f"cannot resolve registry contract for {task!r}: {error}") from error
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
                context_dataset_id_field=registry.descriptor.context_dataset_id_field,
                context_dataset_id_prefix=registry.descriptor.context_dataset_id_prefix,
            )
        except Exception as error:
            raise PreflightError(
                f"cannot resolve semantic dataset identity for {task!r}: {error}"
            ) from error
        description = str(competition.description)
        targets.append(
            TaskTarget(
                task_id=task,
                dataset_id=dataset_id,
                public_dir=public_dir,
                sample_submission_path=(
                    Path(sample).resolve() if isinstance(sample, Path) else None
                ),
                description_text=description,
                description_sha256=sha256(description.encode("utf-8")),
            )
        )
    return targets


def attest_source_public_trees(
    targets: Sequence[TaskTarget],
) -> dict[str, dict[str, Any]]:
    from dslighting.core.task_context import attest_public_tree

    attestations: dict[str, dict[str, Any]] = {}
    for target in targets:
        try:
            attestations[target.task_id] = attest_public_tree(target.public_dir).as_dict()
        except Exception as error:
            raise PreflightError(
                f"cannot attest public data for {target.task_id!r}: {error}"
            ) from error
    return attestations


def resolve_benchmark(source_id: str, data_root: Path) -> tuple[Any, Any, Any, PreparedBenchmark]:
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
    probe_root = resolve_directory(data_root, "benchmark data root")
    try:
        probe_registry = catalog.build_registry(descriptor, data_root=probe_root)
        available = probe_registry.list_competition_ids()
    except Exception as error:
        raise PreflightError(
            f"cannot initialize benchmark registry for {descriptor.source_id!r}: {error}"
        ) from error
    normalized = normalize_data_root(probe_root, available)
    vendor_dir = resolve_directory(descriptor.registry_root, "benchmark registry")
    return catalog, descriptor, profile, PreparedBenchmark(
        descriptor=descriptor,
        data_root=normalized,
        vendor_dir=vendor_dir,
        targets=(),
        source_public_attestations={},
    )


def prepare_benchmark(
    source_id: str,
    data_root: Path,
    *,
    inline_tasks: Sequence[str] | None,
    tasks_file: Path | None,
    limit: int | None,
) -> tuple[Any, PreparedBenchmark]:
    catalog, descriptor, profile, base = resolve_benchmark(source_id, data_root)
    registry = catalog.build_registry(descriptor, data_root=base.data_root)
    targets = tuple(
        resolve_tasks(
            registry=registry,
            inline_tasks=inline_tasks,
            tasks_file=tasks_file,
            limit=limit,
        )
    )
    return profile, PreparedBenchmark(
        descriptor=descriptor,
        data_root=base.data_root,
        vendor_dir=base.vendor_dir,
        targets=targets,
        source_public_attestations=attest_source_public_trees(targets),
    )


def preflight_l1(
    l1_artifact_dir: Path,
    targets: Sequence[TaskTarget],
) -> dict[str, Any]:
    from dslighting.core.task_context import load_l1_artifact, validate_l1_public_coverage

    artifact_dir = resolve_directory(l1_artifact_dir, "L1 artifact directory")
    bundle = hashlib.sha256()
    files: dict[str, dict[str, Any]] = {}
    artifact_digests: dict[str, tuple[Path, str]] = {}
    for target in targets:
        artifact_id = target.dataset_id
        path = artifact_dir / f"{artifact_id}.json"
        if artifact_id not in artifact_digests:
            payload = read_bytes(path, f"L1 artifact for dataset {artifact_id}")
            digest = sha256(payload)
            artifact_digests[artifact_id] = (path.resolve(), digest)
            bundle.update(artifact_id.encode("utf-8"))
            bundle.update(b"\0")
            bundle.update(digest.encode("ascii"))
            bundle.update(b"\n")
        try:
            artifact = load_l1_artifact(
                artifact_dir,
                artifact_id,
                expected_dataset_id=artifact_id,
                expected_task_id=target.task_id,
            )
            excluded = (
                [target.sample_submission_path]
                if target.sample_submission_path is not None
                else []
            )
            public_attestation = validate_l1_public_coverage(
                artifact,
                target.public_dir,
                excluded_paths=excluded,
            )
            rendered = artifact.render_markdown(heading_level=3)
            treated = artifact.replace_dataset_description(target.description_text)
        except Exception as error:
            raise PreflightError(f"invalid L1 artifact {path}: {error}") from error
        resolved_path, digest = artifact_digests[artifact_id]
        files[target.task_id] = {
            "artifact_id": artifact_id,
            "path": str(resolved_path),
            "sha256": digest,
            "source_description_sha256": target.description_sha256,
            "rendered_l1_sha256": sha256(rendered.encode("utf-8")),
            "rendered_l1_chars": len(rendered),
            "treated_description_sha256": sha256(treated.encode("utf-8")),
            "semantic_public_tree": public_attestation.as_dict(),
        }
    return {
        "directory": str(artifact_dir),
        "validated_task_count": len(targets),
        "artifact_count": len(artifact_digests),
        "bundle_sha256": bundle.hexdigest(),
        "files": files,
    }


def preflight_l2(path: Path) -> tuple[Path, dict[str, Any]]:
    from dslighting.core.task_context import load_l2_artifact

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in {".md", ".markdown"}:
        raise PreflightError(f"L2 guidance must be a Markdown file: {resolved}")
    payload = read_bytes(resolved, "L2 guidance")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise PreflightError(f"L2 guidance is not valid UTF-8: {resolved}") from error
    if not text.strip():
        raise PreflightError(f"L2 guidance contains no text: {resolved}")
    try:
        load_l2_artifact(resolved)
    except Exception as error:
        raise PreflightError(f"invalid L2 guidance {resolved}: {error}") from error
    return resolved, {"path": str(resolved), "sha256": sha256(payload)}


def fixed_config_fingerprint(config: Any) -> str:
    payload = config.model_dump(mode="json")
    run = payload.get("run")
    if isinstance(run, dict):
        run.pop("run_name", None)
    task_context = payload.get("task_context")
    if isinstance(task_context, dict):
        for field_name in ("policy", "l1_artifact_dir", "l2_guidance_path"):
            task_context.pop(field_name, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(canonical)


def build_configs(
    *,
    conditions: Sequence[ExperimentCondition],
    run_id: str,
    runtime: ConditionRuntime,
    config_overrides: Mapping[str, Mapping[str, Any]] | None,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    from dslighting.core.config.builder import ConfigBuilder

    configs: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for condition in conditions:
        build_kwargs = {key: dict(value) for key, value in (config_overrides or {}).items()}
        if runtime.sandbox_timeout_seconds is not None:
            sandbox = build_kwargs.setdefault("sandbox", {})
            sandbox.update(
                {
                    "timeout": runtime.sandbox_timeout_seconds,
                    "local_isolation": runtime.sandbox_backend,
                    "environment_policy": runtime.sandbox_environment_policy,
                    "network_policy": runtime.sandbox_network_policy,
                }
            )
        try:
            config = ConfigBuilder().build_config(
                workflow=runtime.workflow,
                model=runtime.model,
                task_context={
                    "policy": condition.task_context_policy,
                    "l1_artifact_dir": (
                        str(condition.l1_artifact_dir)
                        if condition.l1_artifact_dir is not None
                        else None
                    ),
                    "l2_guidance_path": (
                        str(condition.l2_guidance_path)
                        if condition.l2_guidance_path is not None
                        else None
                    ),
                    "require_canonical_layout": True,
                },
                run_name=f"{run_id}_{condition.condition_id}",
                **build_kwargs,
            )
        except Exception as error:
            raise PreflightError(
                f"cannot build config for condition {condition.condition_id!r}: {error}"
            ) from error
        config.scheduler.max_concurrency = runtime.task_concurrency
        config.scheduler.run_id = run_id
        config.run.parameters = dict(config.run.parameters)
        config.run.parameters["output_artifact_suffix_seed"] = run_id
        if runtime.llm_global_concurrency is not None:
            config.scheduler.llm_max_concurrency = runtime.llm_global_concurrency
        if runtime.device_admission is not None:
            config.scheduler.gpu_policy = runtime.device_admission
        if runtime.checkpoint_resume_enabled is not None:
            config.scheduler.checkpoint_resume_enabled = runtime.checkpoint_resume_enabled
        if runtime.llm_per_key_concurrency is not None:
            config.llm.max_concurrent_per_key = runtime.llm_per_key_concurrency
        if runtime.llm_max_retries is not None:
            config.llm.max_retries = runtime.llm_max_retries
        if runtime.llm_thinking is not None:
            config.llm.thinking = runtime.llm_thinking
        if not dry_run and not config.llm.get_api_keys():
            raise PreflightError(
                "no LLM API key resolved; configure API_KEY/OPENAI_API_KEY or "
                "LLM_MODEL_CONFIGS before a non-dry run"
            )
        configs[condition.condition_id] = config
        fingerprints[condition.condition_id] = fixed_config_fingerprint(config)
    if not configs:
        raise PreflightError("experiment condition selection is empty")
    if len(set(fingerprints.values())) != 1:
        raise PreflightError(
            "conditions changed fixed runtime configuration: "
            + ", ".join(f"{key}={value}" for key, value in fingerprints.items())
        )
    return configs, next(iter(fingerprints.values()))


def result_count(result: Any) -> int | None:
    results = getattr(result, "results", None)
    if results is None:
        return None
    try:
        return len(results)
    except TypeError:
        return None


def result_path(result: Any, field_name: str) -> str | None:
    value = getattr(result, field_name, None)
    return str(value) if value is not None else None


def collect_task_context_audit(
    benchmark: Any,
    *,
    tasks: Sequence[str],
    policy: str,
) -> dict[str, dict[str, Any]]:
    runner = getattr(benchmark, "runner", None)
    get_records = getattr(runner, "get_run_records", None)
    if not callable(get_records):
        raise ExperimentInvariantError(f"condition {policy!r} did not expose task run records")
    expected = set(tasks)
    audits: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    for record in get_records():
        if not isinstance(record, Mapping):
            continue
        task_id = str(record.get("task_id") or "")
        if task_id not in expected:
            continue
        summary = record.get("summary")
        if not isinstance(summary, Mapping) or not isinstance(summary.get("success"), bool):
            raise ExperimentInvariantError(
                f"condition {policy!r} task {task_id!r} has no boolean run outcome"
            )
        attempts[task_id] = attempts.get(task_id, 0) + 1
        summaries[task_id] = {"success": summary["success"], "result": summary.get("result")}
        runtime_audit = record.get("task_context_audit")
        provenance = runtime_audit.get("provenance") if isinstance(runtime_audit, Mapping) else None
        fixed = runtime_audit.get("fixed_context") if isinstance(runtime_audit, Mapping) else None
        if not isinstance(provenance, Mapping) or not isinstance(fixed, Mapping):
            raise ExperimentInvariantError(
                f"condition {policy!r} task {task_id!r} did not persist task-context audit"
            )
        if provenance.get("policy") != policy:
            raise ExperimentInvariantError(
                f"condition {policy!r} task {task_id!r} recorded policy "
                f"{provenance.get('policy')!r}"
            )
        dataset_id = provenance.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ExperimentInvariantError(
                f"condition {policy!r} task {task_id!r} recorded no dataset identity"
            )
        expects_l1 = policy in {"l1", "l3"}
        expects_l2 = policy in {"l2", "l3"}
        if (provenance.get("l1") is not None) != expects_l1 or (
            provenance.get("l2") is not None
        ) != expects_l2:
            raise ExperimentInvariantError(
                f"condition {policy!r} task {task_id!r} violates its L1/L2 treatment"
            )
        context = {
            "task_context_provenance": dict(provenance),
            "fixed_task_context": dict(fixed),
        }
        previous = audits.get(task_id)
        if previous is not None and previous != context:
            raise ExperimentInvariantError(
                f"condition {policy!r} produced conflicting retry records for {task_id!r}"
            )
        audits[task_id] = context
    missing = sorted(expected - audits.keys())
    if missing:
        raise ExperimentInvariantError(
            f"condition {policy!r} has no auditable context for: {', '.join(missing)}"
        )
    return {
        task: {
            **audits[task],
            "run_summary": {**summaries[task], "attempt_count": attempts[task]},
        }
        for task in tasks
    }


def collect_task_outcomes(
    benchmark: Any,
    audit_by_task: Mapping[str, Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    condition_id: str,
) -> dict[str, dict[str, Any]]:
    get_columns = getattr(benchmark, "get_result_columns", None)
    rows = getattr(benchmark, "results", None)
    if not callable(get_columns) or not isinstance(rows, list):
        raise ExperimentInvariantError(
            f"condition {condition_id!r} did not expose benchmark outcomes"
        )
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
            f"condition {condition_id!r} exposed an incompatible result schema"
        )
    expected = set(tasks)
    outcomes: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != len(columns):
            raise ExperimentInvariantError(
                f"condition {condition_id!r} exposed a malformed result row"
            )
        row = dict(zip(columns, raw_row))
        task_id = str(row.get("competition_id") or "")
        if task_id not in expected:
            continue
        if task_id in outcomes:
            raise ExperimentInvariantError(
                f"condition {condition_id!r} produced duplicate outcome for {task_id!r}"
            )
        submission_exists = row.get("submission_exists")
        valid_submission = row.get("valid_submission")
        if not isinstance(submission_exists, bool) or not isinstance(valid_submission, bool):
            raise ExperimentInvariantError(
                f"condition {condition_id!r} task {task_id!r} has invalid submission status"
            )
        run_summary = audit_by_task[task_id].get("run_summary")
        if not isinstance(run_summary, Mapping) or not isinstance(
            run_summary.get("success"), bool
        ):
            raise ExperimentInvariantError(
                f"condition {condition_id!r} task {task_id!r} has no workflow outcome"
            )
        workflow_success = run_summary["success"]
        status = (
            "workflow_failed"
            if not workflow_success
            else "completed"
            if valid_submission
            else "invalid_submission"
        )
        outcomes[task_id] = {
            "status": status,
            "workflow_success": workflow_success,
            "submission_exists": submission_exists,
            "valid_submission": valid_submission,
            "score": row.get("score"),
            "error_message": row.get("error_message"),
        }
    missing = sorted(expected - outcomes.keys())
    if missing:
        raise ExperimentInvariantError(
            f"condition {condition_id!r} has no outcome for: {', '.join(missing)}"
        )
    return {task: outcomes[task] for task in tasks}


def verify_fixed_context(
    audit_by_task: Mapping[str, Mapping[str, Any]],
    reference: dict[str, dict[str, str]],
    *,
    condition_id: str,
) -> None:
    for task_id, context_record in audit_by_task.items():
        provenance = context_record.get("task_context_provenance")
        fixed = context_record.get("fixed_task_context")
        if not isinstance(provenance, Mapping) or not isinstance(fixed, Mapping):
            raise ExperimentInvariantError(
                f"condition {condition_id!r} task {task_id!r} has malformed audit data"
            )
        values = {
            "main_data_report_sha256": provenance.get("main_data_report_sha256"),
            "io_instructions_sha256": fixed.get("io_instructions_sha256"),
            "output_artifact_name": fixed.get("output_artifact_name"),
            "submission_contract_sha256": fixed.get("submission_contract_sha256"),
            "evaluation_contract_ref_sha256": fixed.get("evaluation_contract_ref_sha256"),
        }
        fingerprint: dict[str, str] = {}
        for field_name in FIXED_CONTEXT_FIELDS:
            value = values[field_name]
            if not isinstance(value, str) or not value:
                raise ExperimentInvariantError(
                    f"condition {condition_id!r} task {task_id!r} has no {field_name}"
                )
            fingerprint[field_name] = value
        expected = reference.setdefault(task_id, fingerprint)
        if fingerprint != expected:
            changed = [
                field_name
                for field_name in FIXED_CONTEXT_FIELDS
                if fingerprint[field_name] != expected[field_name]
            ]
            raise ExperimentInvariantError(
                f"condition {condition_id!r} changed fixed context for {task_id!r}: "
                + ", ".join(changed)
            )


def verify_selected_artifacts(
    audit_by_task: Mapping[str, Mapping[str, Any]],
    *,
    condition: ExperimentCondition,
    l1_summary: Mapping[str, Any] | None,
    l2_summary: Mapping[str, Any] | None,
) -> None:
    for task_id, context_record in audit_by_task.items():
        provenance = context_record.get("task_context_provenance")
        if not isinstance(provenance, Mapping):
            raise ExperimentInvariantError(
                f"condition {condition.condition_id!r} task {task_id!r} has no provenance"
            )
        if condition.task_context_policy in {"l1", "l3"}:
            files = l1_summary.get("files") if isinstance(l1_summary, Mapping) else None
            expected = files.get(task_id) if isinstance(files, Mapping) else None
            expected_provenance = (
                {"path": expected.get("path"), "sha256": expected.get("sha256")}
                if isinstance(expected, Mapping)
                else None
            )
            if (
                expected_provenance is None
                or provenance.get("l1") != expected_provenance
                or provenance.get("dataset_id") != expected.get("artifact_id")
            ):
                raise ExperimentInvariantError(
                    f"condition {condition.condition_id!r} task {task_id!r} "
                    "did not use the preflighted L1 artifact"
                )
        if condition.task_context_policy in {"l2", "l3"}:
            expected = (
                {"path": l2_summary.get("path"), "sha256": l2_summary.get("sha256")}
                if isinstance(l2_summary, Mapping)
                else None
            )
            if expected is None or provenance.get("l2") != expected:
                raise ExperimentInvariantError(
                    f"condition {condition.condition_id!r} task {task_id!r} "
                    "did not use the preflighted L2 artifact"
                )


def benchmark_manifest(prepared: PreparedBenchmark) -> dict[str, Any]:
    descriptor = prepared.descriptor
    source_manifest = None
    if descriptor.manifest_path is not None:
        payload = read_bytes(descriptor.manifest_path, "benchmark source manifest")
        source_manifest = {
            "path": str(descriptor.manifest_path),
            "sha256": sha256(payload),
        }
    return {
        "source_id": descriptor.source_id,
        "contract_id": descriptor.contract_id,
        "engine_id": descriptor.engine_id,
        "data_root": str(prepared.data_root),
        "registry_root": str(prepared.vendor_dir),
        "source_manifest": source_manifest,
    }


def selection_manifest(prepared: PreparedBenchmark) -> dict[str, Any]:
    return {
        "task_count": len(prepared.targets),
        "tasks": list(prepared.tasks),
        "dataset_ids": {
            target.task_id: target.dataset_id for target in prepared.targets
        },
        "public_dirs": {
            target.task_id: str(target.public_dir) for target in prepared.targets
        },
        "descriptions": {
            target.task_id: {
                "sha256": target.description_sha256,
                "chars": len(target.description_text),
            }
            for target in prepared.targets
        },
        "source_public_trees": dict(prepared.source_public_attestations),
    }


class ConditionExperimentEngine:
    """Execute repetitions and conditions through the native DSBenchmark path."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.store = ManifestStore(self.run_root)

    def execute(
        self,
        *,
        manifest: dict[str, Any],
        prepared: PreparedBenchmark,
        conditions: Sequence[ExperimentCondition],
        runs: Sequence[EngineRun],
        resume: bool,
        hooks: EngineHooks = EngineHooks(),
    ) -> Path:
        from dslighting.api.benchmark import DSBenchmark

        if not runs:
            raise PreflightError("experiment has no repetitions")
        condition_by_id = {condition.condition_id: condition for condition in conditions}
        if len(condition_by_id) != len(conditions):
            raise PreflightError("experiment condition IDs are not unique")
        tasks = list(prepared.tasks)
        records = manifest.setdefault("runs", [])
        if not isinstance(records, list):
            raise PreflightError("manifest runs must be a list")
        records_by_key: dict[tuple[int, str], dict[str, Any]] = {}
        for raw in records:
            if not isinstance(raw, dict):
                raise PreflightError("manifest run record must be an object")
            key = (raw.get("repetition"), raw.get("condition_id"))
            if not isinstance(key[0], int) or key[1] not in condition_by_id:
                raise PreflightError(f"manifest has an unexpected run record: {key!r}")
            if key in records_by_key:
                raise PreflightError(f"manifest repeats run record {key!r}")
            records_by_key[key] = raw

        references = manifest.setdefault("context_audit", {}).setdefault(
            "reference_by_repetition", {}
        )
        if not isinstance(references, dict):
            raise PreflightError("manifest context references are malformed")
        completed_with_task_failures = False
        manifest["status"] = "running"
        self.store.write(manifest)

        for engine_run in runs:
            reference = references.setdefault(str(engine_run.repetition), {})
            if not isinstance(reference, dict):
                raise PreflightError("manifest repetition context reference is malformed")
            for condition in conditions:
                condition_id = condition.condition_id
                key = (engine_run.repetition, condition_id)
                record = records_by_key.get(key)
                log_path = (
                    self.run_root
                    / f"repetition-{engine_run.repetition:02d}"
                    / condition_id
                )
                if record is not None and record.get("status") in COMPLETED_STATUSES:
                    self._verify_completed(
                        record=record,
                        log_path=log_path,
                        tasks=tasks,
                        condition=condition,
                        l1_summary=engine_run.l1_summaries.get(condition_id),
                        l2_summary=engine_run.l2_summaries.get(condition_id),
                        reference=reference,
                        scoring_inputs=hooks.scoring_inputs,
                    )
                    completed_with_task_failures |= (
                        record.get("status") == "completed_with_task_failures"
                    )
                    continue
                if record is not None and not resume:
                    raise PreflightError(f"run record already exists for {key!r}")
                if record is None:
                    record = {}
                    records.append(record)
                    records_by_key[key] = record
                else:
                    attempts = record.setdefault("resume_attempts", [])
                    if not isinstance(attempts, list):
                        raise PreflightError(f"resume attempts are malformed for {key!r}")
                    attempts.append(
                        {
                            "previous_status": record.get("status"),
                            "previous_error_type": record.get("error_type"),
                            "previous_error": record.get("error"),
                            "archived_log_path": self.store.archive_log(
                                log_path,
                                f"r{engine_run.repetition:02d}-{condition_id}",
                            ),
                            "resumed_at_utc": utc_now(),
                            "mode": "full_condition_rerun",
                        }
                    )
                self._reset_record(
                    record,
                    repetition=engine_run.repetition,
                    condition=condition,
                    log_path=log_path,
                    config=engine_run.configs[condition_id],
                )
                self.store.write(manifest)
                try:
                    before = (
                        hooks.scoring_inputs("before_condition", condition_id)
                        if hooks.scoring_inputs is not None
                        else None
                    )
                    if before is not None:
                        record["scoring_input_attestation"] = {
                            "before": dict(before),
                            "after": None,
                        }
                        self.store.write(manifest)
                    result = DSBenchmark(
                        benchmark_type=prepared.descriptor.source_id,
                        exp_name=engine_run.configs[condition_id].run.run_name,
                        data_dir=str(prepared.data_root),
                        vendor_comp_dir=str(prepared.vendor_dir),
                        competitions=tasks,
                    ).run(
                        config=engine_run.configs[condition_id],
                        log_path=str(log_path),
                        verbose=True,
                    )
                    audit = collect_task_context_audit(
                        result,
                        tasks=tasks,
                        policy=condition.task_context_policy,
                    )
                    outcomes = collect_task_outcomes(
                        result,
                        audit,
                        tasks=tasks,
                        condition_id=condition_id,
                    )
                    verify_selected_artifacts(
                        audit,
                        condition=condition,
                        l1_summary=engine_run.l1_summaries.get(condition_id),
                        l2_summary=engine_run.l2_summaries.get(condition_id),
                    )
                    verify_fixed_context(
                        audit,
                        reference,
                        condition_id=condition_id,
                    )
                    after = (
                        hooks.scoring_inputs("after_condition", condition_id)
                        if hooks.scoring_inputs is not None
                        else None
                    )
                    if after is not None:
                        record["scoring_input_attestation"]["after"] = dict(after)
                except Exception as error:
                    self._fail(manifest, record, error, references)
                    raise
                has_failures = any(item["status"] != "completed" for item in outcomes.values())
                completed_with_task_failures |= has_failures
                record.update(
                    {
                        "status": (
                            "completed_with_task_failures" if has_failures else "completed"
                        ),
                        "result_count": result_count(result),
                        "results_path": result_path(result, "results_path"),
                        "metadata_path": result_path(result, "metadata_path"),
                        "task_context_audit": audit,
                        "task_outcomes": outcomes,
                        "completed_at_utc": utc_now(),
                    }
                )
                manifest["context_audit"].update(
                    {"fixed_fields": list(FIXED_CONTEXT_FIELDS), "status": "verified_so_far"}
                )
                self.store.write(manifest)
            if hooks.source_guard is not None:
                hooks.source_guard()

        manifest["status"] = (
            "completed_with_task_failures" if completed_with_task_failures else "completed"
        )
        manifest["context_audit"].update(
            {"fixed_fields": list(FIXED_CONTEXT_FIELDS), "status": "verified"}
        )
        manifest["completed_at_utc"] = utc_now()
        self.store.write(manifest)
        return self.store.path

    def _verify_completed(
        self,
        *,
        record: Mapping[str, Any],
        log_path: Path,
        tasks: Sequence[str],
        condition: ExperimentCondition,
        l1_summary: Mapping[str, Any] | None,
        l2_summary: Mapping[str, Any] | None,
        reference: dict[str, dict[str, str]],
        scoring_inputs: Callable[[str, str], Mapping[str, Any] | None] | None,
    ) -> None:
        audit = record.get("task_context_audit")
        outcomes = record.get("task_outcomes")
        if not isinstance(audit, Mapping) or set(audit) != set(tasks):
            raise PreflightError(
                f"completed condition {condition.condition_id!r} has incomplete audit"
            )
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(tasks):
            raise PreflightError(
                f"completed condition {condition.condition_id!r} has incomplete outcomes"
            )
        for field_name in ("results_path", "metadata_path"):
            value = record.get(field_name)
            if not isinstance(value, str) or not Path(value).is_file():
                raise PreflightError(
                    f"completed condition {condition.condition_id!r} has no {field_name}"
                )
            try:
                Path(value).resolve().relative_to(log_path.resolve())
            except ValueError as error:
                raise PreflightError(
                    f"completed condition {condition.condition_id!r} {field_name} escapes its log"
                ) from error
        verify_selected_artifacts(
            audit,
            condition=condition,
            l1_summary=l1_summary,
            l2_summary=l2_summary,
        )
        verify_fixed_context(audit, reference, condition_id=condition.condition_id)
        if scoring_inputs is not None:
            current = scoring_inputs("resume_completed_condition", condition.condition_id)
            recorded = record.get("scoring_input_attestation")
            if current is not None and (
                not isinstance(recorded, Mapping)
                or recorded.get("before") != current
                or recorded.get("after") != current
            ):
                raise PreflightError(
                    f"completed condition {condition.condition_id!r} scoring inputs changed"
                )

    @staticmethod
    def _reset_record(
        record: dict[str, Any],
        *,
        repetition: int,
        condition: ExperimentCondition,
        log_path: Path,
        config: Any,
    ) -> None:
        attempts = record.get("resume_attempts")
        record.clear()
        if attempts:
            record["resume_attempts"] = attempts
        record.update(
            {
                "repetition": repetition,
                **condition.as_manifest(),
                "status": "running",
                "log_path": str(log_path),
                "native_run_id": config.run.run_name,
                "checkpoint_resume_enabled": config.scheduler.checkpoint_resume_enabled,
                "started_at_utc": utc_now(),
            }
        )

    def _fail(
        self,
        manifest: dict[str, Any],
        record: dict[str, Any],
        error: Exception,
        references: Mapping[str, Any],
    ) -> None:
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": utc_now(),
            }
        )
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": utc_now(),
            }
        )
        manifest["context_audit"] = {
            "fixed_fields": list(FIXED_CONTEXT_FIELDS),
            "status": "failed",
            "reference_by_repetition": dict(references),
        }
        self.store.write(manifest)
