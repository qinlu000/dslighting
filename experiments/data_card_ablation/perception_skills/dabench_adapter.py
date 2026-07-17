"""DABench task-family resolution for perception-skills annotations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.data_card_ablation.engine import (
    PreflightError,
    read_bytes,
    resolve_directory,
    sha256,
)

from .adapters import (
    BenchmarkAdapter,
    require_directory,
    require_file,
)
from .profile import PerceptionSkillCondition

DATASET_FAMILY_MANIFEST_SCHEMA_VERSION = "dabench_perception_dataset_manifest_v1"


@dataclass(frozen=True)
class DABenchFamilyInputs:
    task_to_family: dict[str, str]
    manifest_summary: dict[str, Any]
    canonical_l1_summaries: dict[str, Any]


def _strict_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    payload = read_bytes(path, label)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be a JSON object: {path}")
    return value, payload


def _safe_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise PreflightError(f"{label} must be a safe basename")
    return value


def _load_dataset_family_manifest(
    manifest_path: Path,
    targets: Sequence[Any],
    *,
    expected_clean_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = manifest_path.expanduser().resolve()
    document, _ = _strict_json(manifest, "DABench dataset-family manifest")
    if document.get("schema_version") != DATASET_FAMILY_MANIFEST_SCHEMA_VERSION:
        raise PreflightError("DABench dataset-family manifest schema_version mismatch")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PreflightError("DABench family manifest tasks must be an array")
    task_entries: dict[str, str] = {}
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"tasks[{index}] must be an object")
        task_id = _safe_id(raw.get("task_id"), f"tasks[{index}].task_id")
        family_id = _safe_id(raw.get("family_id"), f"tasks[{index}].family_id")
        if task_id in task_entries:
            raise PreflightError(f"duplicate DABench task in family manifest: {task_id}")
        task_entries[task_id] = family_id

    task_to_family: dict[str, str] = {}
    for target in targets:
        task_id = target.task_id
        family_id = task_entries.get(task_id)
        if family_id is None:
            raise PreflightError(f"selected task {task_id!r} is missing from family manifest")
        if target.dataset_id != task_id:
            raise PreflightError("DABench family adaptation requires task-scoped identity")
        task_to_family[task_id] = family_id
    return task_to_family, {
        "path": str(manifest),
        "schema_version": DATASET_FAMILY_MANIFEST_SCHEMA_VERSION,
        "clean_root": str(expected_clean_root.resolve()) if expected_clean_root else None,
        "declared_task_count": len(raw_tasks),
        "selected_task_count": len(task_to_family),
        "selected_family_count": len(set(task_to_family.values())),
        "task_to_family": task_to_family,
    }


def _preflight_family_l1(
    annotations_dir: Path,
    targets: Sequence[Any],
    task_to_family: Mapping[str, str],
) -> dict[str, Any]:
    from dslighting.core.task_context import load_l1_artifact, validate_l1_public_coverage

    root = resolve_directory(annotations_dir, "canonical family L1 annotations")
    canonical_files: dict[str, dict[str, Any]] = {}
    task_files: dict[str, dict[str, Any]] = {}
    bundle = hashlib.sha256()
    for target in targets:
        family_id = task_to_family.get(target.task_id)
        if family_id is None:
            raise PreflightError(f"no dataset family resolved for {target.task_id!r}")
        path = root / f"{family_id}.json"
        if family_id not in canonical_files:
            document, payload = _strict_json(path, f"canonical L1 for {family_id}")
            if document.get("dataset_id") != family_id or "task_id" in document:
                raise PreflightError(f"canonical L1 identity is invalid: {path}")
            try:
                artifact = load_l1_artifact(root, family_id, expected_dataset_id=family_id)
            except Exception as error:
                raise PreflightError(f"invalid canonical L1 artifact {path}: {error}") from error
            digest = sha256(payload)
            rendered = artifact.render_markdown(heading_level=3)
            canonical_files[family_id] = {
                "path": str(path.resolve()),
                "sha256": digest,
                "rendered_l1_sha256": sha256(rendered.encode("utf-8")),
                "rendered_l1_chars": len(rendered),
            }
            bundle.update(f"{family_id}\0{digest}\n".encode())
        try:
            artifact = load_l1_artifact(root, family_id, expected_dataset_id=family_id)
            public = validate_l1_public_coverage(
                artifact,
                target.public_dir,
                excluded_paths=(
                    [target.sample_submission_path]
                    if target.sample_submission_path is not None
                    else []
                ),
            )
        except Exception as error:
            raise PreflightError(
                f"invalid canonical L1 for task {target.task_id!r}: {error}"
            ) from error
        item = canonical_files[family_id]
        task_files[target.task_id] = {
            "family_id": family_id,
            "canonical_path": item["path"],
            "canonical_sha256": item["sha256"],
            "rendered_l1_sha256": item["rendered_l1_sha256"],
            "semantic_public_tree": public.as_dict(),
        }
    return {
        "directory": str(root),
        "validated_task_count": len(targets),
        "artifact_count": len(canonical_files),
        "bundle_sha256": bundle.hexdigest(),
        "canonical_files": canonical_files,
        "tasks": task_files,
    }


def _materialize_family_annotations(
    skills: Sequence[PerceptionSkillCondition],
    *,
    targets: Sequence[Any],
    task_to_family: Mapping[str, str],
    run_root: Path,
) -> tuple[tuple[PerceptionSkillCondition, ...], dict[str, Any]]:
    from dslighting.core.task_context import load_l1_artifact

    resolved_root = run_root / "resolved_annotations"
    runtime_skills: list[PerceptionSkillCondition] = []
    summaries: dict[str, Any] = {}
    for skill in skills:
        condition_id = skill.perception_skill_id
        condition_dir = resolved_root / condition_id
        condition_dir.mkdir(parents=True, exist_ok=False)
        files: dict[str, Any] = {}
        for target in targets:
            family_id = task_to_family[target.task_id]
            source = skill.annotations_dir / f"{family_id}.json"
            document, _ = _strict_json(source, f"canonical L1 for {family_id}")
            resolved_document = {
                **document,
                "dataset_id": target.task_id,
                "task_id": target.task_id,
            }
            resolved_payload = (
                json.dumps(resolved_document, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            destination = condition_dir / f"{target.task_id}.json"
            try:
                destination.write_bytes(resolved_payload)
            except OSError as error:
                raise PreflightError(f"cannot write resolved L1 {destination}: {error}") from error
            try:
                load_l1_artifact(
                    condition_dir,
                    target.task_id,
                    expected_dataset_id=target.task_id,
                    expected_task_id=target.task_id,
                )
            except Exception as error:
                raise PreflightError(f"invalid resolved L1 {destination}: {error}") from error
            files[target.task_id] = {
                "family_id": family_id,
                "resolved_sha256": sha256(resolved_payload),
            }
        runtime_skills.append(
            PerceptionSkillCondition(condition_id, skill.skill_entrypoints, condition_dir.resolve())
        )
        summaries[condition_id] = {"resolved_directory": str(condition_dir), "files": files}
    return tuple(runtime_skills), {"root": str(resolved_root), "conditions": summaries}


def _runtime_skills_at(
    skills: Sequence[PerceptionSkillCondition],
    run_root: Path,
) -> tuple[PerceptionSkillCondition, ...]:
    return tuple(
        PerceptionSkillCondition(
            skill.perception_skill_id,
            skill.skill_entrypoints,
            (run_root / "resolved_annotations" / skill.perception_skill_id).resolve(),
        )
        for skill in skills
    )


class DABenchAdapter(BenchmarkAdapter):
    def preflight(self) -> Mapping[str, Any]:
        profile = self.profile
        data_root = require_directory(profile.data_root, "DABench clean release")
        family_manifest = require_file(
            profile.dataset_family_manifest or Path(),
            "DABench family manifest",
        )
        skills_manifest = require_file(profile.skills_manifest, "DABench skills manifest")
        annotations_root = require_directory(
            profile.annotations_root,
            "DABench annotation publication",
        )
        return {
            "benchmark": profile.benchmark,
            "data_root": str(data_root),
            "dataset_family_manifest": str(family_manifest),
            "skills_manifest": str(skills_manifest),
            "annotations_root": str(annotations_root),
        }

    def prepare_family_inputs(
        self,
        *,
        manifest_path: Path,
        targets: Sequence[Any],
        data_root: Path,
        perception_skills: Sequence[PerceptionSkillCondition],
    ) -> DABenchFamilyInputs:
        mapping, summary = _load_dataset_family_manifest(
            manifest_path,
            targets,
            expected_clean_root=data_root,
        )
        canonical = {
            skill.perception_skill_id: _preflight_family_l1(
                skill.annotations_dir,
                targets,
                mapping,
            )
            for skill in perception_skills
        }
        return DABenchFamilyInputs(mapping, summary, canonical)

    def runtime_skills(
        self,
        skills: Sequence[PerceptionSkillCondition],
        run_root: Path,
    ) -> tuple[PerceptionSkillCondition, ...]:
        return _runtime_skills_at(skills, run_root)

    def materialize_family_annotations(
        self,
        skills: Sequence[PerceptionSkillCondition],
        *,
        targets: Sequence[Any],
        family_inputs: DABenchFamilyInputs,
        run_root: Path,
    ) -> tuple[tuple[PerceptionSkillCondition, ...], dict[str, Any]]:
        return _materialize_family_annotations(
            skills,
            targets=targets,
            task_to_family=family_inputs.task_to_family,
            run_root=run_root,
        )
