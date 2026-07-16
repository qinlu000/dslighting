"""Single-layer runner for versioned perception-skills experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.data_card_ablation.engine import (
    FIXED_CONTEXT_FIELDS,
    ConditionExperimentEngine,
    ConditionRuntime,
    EngineHooks,
    EngineRun,
    ExperimentCondition,
    ManifestStore,
    ManifestStoreError,
    PreflightError,
    benchmark_manifest,
    build_configs,
    preflight_l1,
    prepare_benchmark,
    read_bytes,
    selection_manifest,
    sha256,
    utc_now,
)

from .adapters import BenchmarkAdapter, adapter_for
from .dabench_adapter import DABenchAdapter, DABenchFamilyInputs
from .profile import (
    EXPERIMENT_ROOT,
    PROJECT_ROOT,
    ExperimentRequest,
    PerceptionSkillCondition,
    PerceptionSkillsExperimentProfile,
)
from .runtime import DataContainedRuntime

BATCHES_ROOT = PROJECT_ROOT / "data" / "experiments" / "data_card_ablation" / "batches"


class ExperimentConfigurationError(RuntimeError):
    """Raised when a request violates its frozen experiment protocol."""


def _safe_id(value: str) -> bool:
    return bool(
        value
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _strict_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    payload = read_bytes(path, label)

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


def load_perception_skills(
    manifest_path: Path,
    annotations_root: Path,
    selected_ids: Sequence[str],
) -> tuple[tuple[PerceptionSkillCondition, ...], dict[str, Any]]:
    manifest = manifest_path.expanduser().resolve()
    document, payload = _strict_json(manifest, "perception skills manifest")
    if document.get("schema_version") != "perception_skills_v1":
        raise PreflightError("perception skills manifest schema_version mismatch")
    raw_conditions = document.get("perception_skills")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise PreflightError("perception skills manifest must contain a non-empty list")
    annotations = annotations_root.expanduser().resolve()
    if not annotations.is_dir():
        raise PreflightError(f"annotations root is not a directory: {annotations}")
    skills_root = manifest.parent.resolve()
    by_id: dict[str, PerceptionSkillCondition] = {}
    entrypoint_summaries: dict[str, list[dict[str, str]]] = {}
    for index, raw in enumerate(raw_conditions):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"perception_skills[{index}] must be an object")
        if set(raw) != {"perception_skill_id", "skill_entrypoints"}:
            raise PreflightError(f"perception_skills[{index}] has invalid fields")
        skill_id = raw.get("perception_skill_id")
        if not isinstance(skill_id, str) or not _safe_id(skill_id):
            raise PreflightError(
                f"perception_skills[{index}].perception_skill_id must be a safe basename"
            )
        if skill_id in by_id:
            raise PreflightError(f"duplicate perception skill ID: {skill_id}")
        raw_entrypoints = raw.get("skill_entrypoints")
        if not isinstance(raw_entrypoints, list) or not all(
            isinstance(value, str) and value for value in raw_entrypoints
        ):
            raise PreflightError(f"perception skill {skill_id!r} has invalid entrypoints")
        entrypoints: list[Path] = []
        summaries: list[dict[str, str]] = []
        for relative in raw_entrypoints:
            candidate = Path(relative)
            if candidate.is_absolute():
                raise PreflightError(
                    f"perception skill {skill_id!r} entrypoint must be relative"
                )
            entrypoint = _resolve_child(
                skills_root / candidate,
                skills_root,
                f"perception skill {skill_id!r} entrypoint",
            )
            entrypoint_payload = read_bytes(entrypoint, f"skill {skill_id!r} entrypoint")
            entrypoints.append(entrypoint)
            summaries.append({"path": str(entrypoint), "sha256": sha256(entrypoint_payload)})
        annotations_dir = _resolve_child(
            annotations / skill_id,
            annotations,
            f"perception skill {skill_id!r} annotations",
        )
        if not annotations_dir.is_dir():
            raise PreflightError(
                f"perception skill {skill_id!r} annotations are missing: {annotations_dir}"
            )
        by_id[skill_id] = PerceptionSkillCondition(
            skill_id,
            tuple(entrypoints),
            annotations_dir,
        )
        entrypoint_summaries[skill_id] = summaries
    unknown = [skill_id for skill_id in selected_ids if skill_id not in by_id]
    if unknown:
        raise PreflightError("unknown perception skill IDs: " + ", ".join(unknown))
    selected = tuple(by_id[skill_id] for skill_id in selected_ids)
    if not selected:
        raise PreflightError("perception skill selection is empty")
    return selected, {
        "path": str(manifest),
        "sha256": sha256(payload),
        "schema_version": document["schema_version"],
        "selected": [
            {
                "perception_skill_id": skill.perception_skill_id,
                "control": skill.is_control,
                "skill_entrypoints": entrypoint_summaries[skill.perception_skill_id],
                "annotations_dir": str(skill.annotations_dir),
            }
            for skill in selected
        ],
    }


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    files = list((PROJECT_ROOT / "dslighting").rglob("*.py"))
    files.extend(EXPERIMENT_ROOT.glob("*.py"))
    files.extend((EXPERIMENT_ROOT / "perception_skills").glob("*.py"))
    files.extend((EXPERIMENT_ROOT / "skills").rglob("*.json"))
    files.extend((EXPERIMENT_ROOT / "skills").rglob("*.md"))
    for path in sorted({path.resolve() for path in files if path.is_file()}):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _runtime(profile: PerceptionSkillsExperimentProfile) -> ConditionRuntime:
    policy = profile.runtime
    return ConditionRuntime(
        workflow=policy.workflow,
        model=policy.model,
        task_concurrency=policy.task_concurrency,
        llm_global_concurrency=policy.llm_global_concurrency,
        llm_per_key_concurrency=policy.llm_per_key_concurrency,
        llm_max_retries=policy.llm_max_retries,
        llm_thinking=policy.llm_thinking,
        sandbox_timeout_seconds=policy.sandbox_timeout_seconds,
        sandbox_backend=policy.sandbox_backend,
        sandbox_environment_policy=policy.sandbox_environment_policy,
        sandbox_network_policy=policy.sandbox_network_policy,
        device_admission=policy.device_admission,
        checkpoint_resume_enabled=policy.checkpoint_resume_enabled,
    )


def _conditions(
    skills: Sequence[PerceptionSkillCondition],
) -> tuple[ExperimentCondition, ...]:
    return (
        ExperimentCondition(
            condition_id="main",
            condition_type="main_reference",
            task_context_policy="main",
        ),
        *(
            ExperimentCondition(
                condition_id=skill.perception_skill_id,
                condition_type="perception_skill",
                treatment_id=skill.perception_skill_id,
                control=skill.is_control,
                task_context_policy="l1",
                l1_artifact_dir=skill.annotations_dir,
            )
            for skill in skills
        ),
    )


def _fixed_inputs(document: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


class PerceptionSkillsRunner:
    """Validate and execute one profile with a single experiment manifest."""

    def __init__(
        self,
        profile: PerceptionSkillsExperimentProfile,
        *,
        adapter: BenchmarkAdapter | None = None,
        runtime: DataContainedRuntime | None = None,
    ) -> None:
        self.base_profile = profile
        self.adapter = adapter or adapter_for(profile)
        self.runtime_environment = runtime or DataContainedRuntime(PROJECT_ROOT)

    def run(self, request: ExperimentRequest) -> Path | None:
        self._validate_request(request)
        profile, canonical = self._effective_profile(request)
        if profile != self.base_profile:
            self.adapter = adapter_for(profile)
        preflight = self.adapter.preflight()
        batch_id = request.resume_run_id or self._batch_id(profile, dry_run=not request.execute)
        self.runtime_environment.activate(batch_id)
        fingerprint = source_fingerprint()
        benchmark_profile, prepared = prepare_benchmark(
            profile.benchmark,
            profile.data_root,
            inline_tasks=request.selection.tasks or None,
            tasks_file=request.selection.tasks_file,
            limit=request.selection.limit,
        )
        selected_ids = tuple(condition for condition in profile.conditions if condition != "main")
        skills, skills_summary = load_perception_skills(
            profile.skills_manifest,
            profile.annotations_root,
            selected_ids,
        )
        dabench = self.adapter if isinstance(self.adapter, DABenchAdapter) else None
        family_inputs = self._family_inputs(
            dabench,
            profile,
            prepared,
            skills,
        )
        run_root = BATCHES_ROOT / batch_id
        existing: dict[str, Any] | None = None
        if request.resume_run_id is not None:
            try:
                existing = ManifestStore(run_root).read()
            except ManifestStoreError as error:
                raise ExperimentConfigurationError(str(error)) from error
        elif request.execute and (run_root.exists() or run_root.is_symlink()):
            raise ExperimentConfigurationError(f"batch root already exists: {run_root}")

        runtime_skills, l1_summaries = self._runtime_annotations(
            execute=request.execute,
            resume=existing is not None,
            run_root=run_root,
            prepared=prepared,
            skills=skills,
            dabench=dabench,
            family_inputs=family_inputs,
        )
        conditions = _conditions(runtime_skills)
        runtime = _runtime(profile)
        engine_runs: list[EngineRun] = []
        fixed_config_sha256: str | None = None
        for repetition in range(1, request.repetitions + 1):
            repetition_id = f"{batch_id}-r{repetition:02d}"
            configs, config_sha = build_configs(
                conditions=conditions,
                run_id=repetition_id,
                runtime=runtime,
                config_overrides=benchmark_profile.config_overrides,
                dry_run=not request.execute,
            )
            if fixed_config_sha256 is None:
                fixed_config_sha256 = config_sha
            elif config_sha != fixed_config_sha256:
                raise PreflightError("repetitions changed fixed runtime configuration")
            engine_runs.append(
                EngineRun(
                    repetition=repetition,
                    run_id=repetition_id,
                    configs=configs,
                    l1_summaries={
                        condition.condition_id: l1_summaries[condition.condition_id]
                        for condition in conditions
                        if condition.condition_id in l1_summaries
                    },
                )
            )
        assert fixed_config_sha256 is not None
        first_config = engine_runs[0].configs[conditions[0].condition_id]
        fixed_document = {
            "profile": profile.as_manifest(),
            "profile_sha256": profile.sha256,
            "protocol_classification": "canonical" if canonical else "noncanonical",
            "benchmark": benchmark_manifest(prepared),
            "selection": selection_manifest(prepared),
            "perception_skills": skills_summary,
            "canonical_l1_annotations": (
                family_inputs.canonical_l1_summaries if family_inputs is not None else l1_summaries
            ),
            "runtime_l1_annotations": l1_summaries,
            "dataset_family_manifest": (
                family_inputs.manifest_summary if family_inputs is not None else None
            ),
            "fixed_config_sha256": fixed_config_sha256,
            "repetitions": request.repetitions,
        }
        fixed_sha = _fixed_inputs(fixed_document)
        if existing is not None:
            fixed = existing.get("fixed_inputs")
            if not isinstance(fixed, Mapping) or fixed.get("sha256") != fixed_sha:
                raise ExperimentConfigurationError("resume fixed input fingerprint mismatch")
            manifest = existing
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["last_resumed_at_utc"] = utc_now()
        else:
            manifest = self._manifest(
                batch_id=batch_id,
                request=request,
                profile=profile,
                canonical=canonical,
                preflight=preflight,
                fingerprint=fingerprint,
                prepared=prepared,
                conditions=conditions,
                first_config=first_config,
                fixed_config_sha256=fixed_config_sha256,
                fixed_sha=fixed_sha,
                skills_summary=skills_summary,
                l1_summaries=l1_summaries,
                family_inputs=family_inputs,
            )
        if not request.execute:
            manifest["runs"] = [
                {
                    "repetition": repetition,
                    **condition.as_manifest(),
                    "status": "validated",
                }
                for repetition in range(1, request.repetitions + 1)
                for condition in conditions
            ]
            manifest["status"] = "validated"
            manifest["completed_at_utc"] = utc_now()
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return None

        expected_fingerprint = fingerprint

        def guard() -> None:
            if source_fingerprint() != expected_fingerprint:
                raise ExperimentConfigurationError(
                    "experiment source changed during the batch"
                )

        scoring = None
        if dabench is not None and family_inputs is not None:
            def scoring(
                phase: str,
                condition_id: str,
            ) -> Mapping[str, Any] | None:
                return dabench.verify_scoring_inputs(
                    family_inputs.manifest_summary,
                    phase=phase,
                    condition_id=condition_id,
                )
        return ConditionExperimentEngine(run_root).execute(
            manifest=manifest,
            prepared=prepared,
            conditions=conditions,
            runs=engine_runs,
            resume=existing is not None,
            hooks=EngineHooks(scoring_inputs=scoring, source_guard=guard),
        )

    def _family_inputs(
        self,
        adapter: DABenchAdapter | None,
        profile: PerceptionSkillsExperimentProfile,
        prepared: Any,
        skills: Sequence[PerceptionSkillCondition],
    ) -> DABenchFamilyInputs | None:
        if profile.dataset_family_manifest is None:
            return None
        if adapter is None:
            raise PreflightError("dataset-family manifest requires the DABench adapter")
        return adapter.prepare_family_inputs(
            manifest_path=profile.dataset_family_manifest,
            targets=prepared.targets,
            data_root=prepared.data_root,
            vendor_root=prepared.vendor_dir,
            perception_skills=skills,
        )

    def _runtime_annotations(
        self,
        *,
        execute: bool,
        resume: bool,
        run_root: Path,
        prepared: Any,
        skills: Sequence[PerceptionSkillCondition],
        dabench: DABenchAdapter | None,
        family_inputs: DABenchFamilyInputs | None,
    ) -> tuple[
        tuple[PerceptionSkillCondition, ...],
        dict[str, Mapping[str, Any]],
    ]:
        if family_inputs is None:
            summaries = {
                skill.perception_skill_id: preflight_l1(skill.annotations_dir, prepared.targets)
                for skill in skills
            }
            return tuple(skills), summaries
        if not execute:
            return tuple(skills), dict(family_inputs.canonical_l1_summaries)
        if dabench is None:
            raise PreflightError("DABench adapter is unavailable")
        if resume:
            runtime_skills = dabench.runtime_skills(skills, run_root)
        else:
            run_root.mkdir(parents=True, exist_ok=False)
            runtime_skills, _ = dabench.materialize_family_annotations(
                skills,
                targets=prepared.targets,
                family_inputs=family_inputs,
                run_root=run_root,
            )
        summaries = {
            skill.perception_skill_id: preflight_l1(skill.annotations_dir, prepared.targets)
            for skill in runtime_skills
        }
        return runtime_skills, summaries

    def _manifest(
        self,
        *,
        batch_id: str,
        request: ExperimentRequest,
        profile: PerceptionSkillsExperimentProfile,
        canonical: bool,
        preflight: Mapping[str, Any],
        fingerprint: str,
        prepared: Any,
        conditions: Sequence[ExperimentCondition],
        first_config: Any,
        fixed_config_sha256: str,
        fixed_sha: str,
        skills_summary: Mapping[str, Any],
        l1_summaries: Mapping[str, Any],
        family_inputs: DABenchFamilyInputs | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "perception_skills_experiment_v2",
            "experiment": "perception_skills_downstream_ablation",
            "batch_id": batch_id,
            "run_id": batch_id,
            "status": "validated" if not request.execute else "preparing",
            "dry_run": not request.execute,
            "created_at_utc": utc_now(),
            "resume_count": 0,
            "profile": profile.as_manifest(),
            "profile_sha256": profile.sha256,
            "protocol_classification": "canonical" if canonical else "noncanonical",
            "source_fingerprint_sha256": fingerprint,
            "git_head": git_head(),
            "repetitions": request.repetitions,
            "fixed_inputs": {"schema_version": 1, "sha256": fixed_sha},
            "preflight": dict(preflight),
            "benchmark": benchmark_manifest(prepared),
            "selection": selection_manifest(prepared),
            "runtime": {
                "workflow": profile.runtime.workflow,
                "model": profile.runtime.model,
                "concurrency": profile.runtime.task_concurrency,
                "llm_max_concurrency": profile.runtime.llm_global_concurrency,
                "llm_max_concurrent_per_key": profile.runtime.llm_per_key_concurrency,
                "llm_max_retries": profile.runtime.llm_max_retries,
                "llm_thinking": profile.runtime.llm_thinking,
                "gpu_policy": profile.runtime.device_admission,
                "checkpoint_resume_enabled": profile.runtime.checkpoint_resume_enabled,
                "fixed_config_sha256": fixed_config_sha256,
                "output_contract": first_config.output_contract.model_dump(mode="json"),
                "sandbox": first_config.sandbox.model_dump(mode="json"),
            },
            "conditions": [condition.as_manifest() for condition in conditions],
            "main_reference": {
                "condition_id": "main",
                "source": "benchmark_dataset_description",
            },
            "perception_skills": dict(skills_summary),
            "l1_annotations": dict(l1_summaries),
            "dataset_family_adapter": (
                {
                    "manifest": family_inputs.manifest_summary,
                    "task_to_family": family_inputs.task_to_family,
                    "canonical_annotations": family_inputs.canonical_l1_summaries,
                }
                if family_inputs is not None
                else None
            ),
            "context_audit": {
                "fixed_fields": list(FIXED_CONTEXT_FIELDS),
                "status": "not_run" if not request.execute else "pending",
                "reference_by_repetition": {},
            },
            "runs": [],
        }

    def _validate_request(self, request: ExperimentRequest) -> None:
        if request.repetitions < 1:
            raise ExperimentConfigurationError("repetitions must be positive")
        if request.selection.tasks and request.selection.tasks_file is not None:
            raise ExperimentConfigurationError("choose only one of tasks and tasks-file")
        if request.selection.limit is not None and request.selection.limit < 1:
            raise ExperimentConfigurationError("limit must be positive")
        if request.resume_run_id is not None:
            if not request.execute:
                raise ExperimentConfigurationError("resume requires --execute")
            if not _safe_id(request.resume_run_id):
                raise ExperimentConfigurationError("resume run ID must be a safe basename")
        if request.overrides.active and not request.allow_protocol_override:
            raise ExperimentConfigurationError(
                "model/workflow/concurrency overrides require --allow-protocol-override"
            )

    def _effective_profile(
        self,
        request: ExperimentRequest,
    ) -> tuple[PerceptionSkillsExperimentProfile, bool]:
        if not request.overrides.active:
            return self.base_profile, True
        runtime = replace(
            self.base_profile.runtime,
            model=request.overrides.model or self.base_profile.runtime.model,
            workflow=request.overrides.workflow or self.base_profile.runtime.workflow,
            task_concurrency=(
                request.overrides.task_concurrency
                or self.base_profile.runtime.task_concurrency
            ),
        )
        return replace(self.base_profile, runtime=runtime), False

    @staticmethod
    def _batch_id(
        profile: PerceptionSkillsExperimentProfile,
        *,
        dry_run: bool,
    ) -> str:
        suffix = "dry-run" if dry_run else datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        version = profile.profile_id.rsplit("-", 1)[-1]
        return f"{profile.benchmark}-perception-skills-{version}-{suffix}"
