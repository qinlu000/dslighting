"""DABench family-annotation resolution and frozen-input attestation."""

from __future__ import annotations

import hashlib
import json
import os
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
    AdapterPreflightError,
    BenchmarkAdapter,
    require_directory,
    require_file,
    tree_sha256,
)
from .profile import PerceptionSkillCondition

DATASET_FAMILY_MANIFEST_SCHEMA_VERSION = "dabench_perception_dataset_manifest_v1"
DATASET_FAMILY_MANIFEST_FILENAME = "dabench_perception_dataset_manifest.json"
EXPECTED_TASK_COUNT = 257
EXPECTED_FAMILY_COUNT = 51


@dataclass(frozen=True)
class DABenchFamilyInputs:
    task_to_family: dict[str, str]
    manifest_summary: dict[str, Any]
    canonical_l1_summaries: dict[str, Any]


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


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unsupported " + ", ".join(sorted(extra)))
        raise PreflightError(f"{label} has invalid fields: {'; '.join(details)}")


def _safe_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value != value.strip()
    ):
        raise PreflightError(f"{label} must be a safe basename")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _vendor_contract(task_dir: Path, task_id: str) -> tuple[str, list[dict[str, Any]]]:
    if not task_dir.is_dir() or task_dir.is_symlink():
        raise PreflightError(f"{task_id}: vendor task directory is missing or unsafe")
    inventory: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise PreflightError(f"{task_id}: cannot inspect vendor contract: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(task_dir)
            if entry.is_symlink():
                raise PreflightError(
                    f"{task_id}: vendor contract contains symlink {relative.as_posix()!r}"
                )
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise PreflightError(
                    f"{task_id}: vendor contract contains non-regular entry {relative!s}"
                )
            if entry.name.endswith(".pyc"):
                continue
            try:
                payload = path.read_bytes()
                size = entry.stat(follow_symlinks=False).st_size
            except OSError as error:
                raise PreflightError(f"{task_id}: cannot read vendor file {relative}") from error
            inventory.append(
                {"path": relative.as_posix(), "sha256": sha256(payload), "size_bytes": size}
            )
    visit(task_dir)
    inventory.sort(key=lambda item: item["path"])
    return _canonical_sha256(
        {"schema_version": "dabench_vendor_contract_v1", "files": inventory}
    ), inventory


def _attest_file(path: Path, expected: str, label: str) -> str:
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"{label} is missing: {path}") from error
    if canonical != path or path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} path is unsafe: {path}")
    actual = sha256(path.read_bytes())
    if actual != expected:
        raise PreflightError(f"{label} hash changed")
    return actual


def _load_dataset_family_manifest(
    manifest_path: Path,
    targets: Sequence[Any],
    *,
    expected_clean_root: Path | None = None,
    vendor_root: Path | None = None,
    expected_task_count: int | None = None,
    expected_family_count: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = manifest_path.expanduser().resolve()
    document, payload = _strict_json(manifest, "DABench dataset-family manifest")
    _exact_fields(
        document,
        {"schema_version", "source_root", "clean_root", "task_count", "family_count", "tasks", "families"},
        "DABench dataset-family manifest",
    )
    if document.get("schema_version") != DATASET_FAMILY_MANIFEST_SCHEMA_VERSION:
        raise PreflightError("DABench dataset-family manifest schema_version mismatch")
    clean_root = Path(str(document.get("clean_root"))).expanduser().resolve()
    if expected_clean_root is not None and clean_root != expected_clean_root.resolve():
        raise PreflightError("DABench manifest clean_root does not match benchmark data root")
    if manifest != clean_root / DATASET_FAMILY_MANIFEST_FILENAME:
        raise PreflightError("DABench family manifest is not at its canonical release path")
    raw_tasks = document.get("tasks")
    raw_families = document.get("families")
    if not isinstance(raw_tasks, list) or not isinstance(raw_families, list):
        raise PreflightError("DABench family manifest tasks/families must be arrays")
    if document.get("task_count") != len(raw_tasks) or document.get("family_count") != len(
        raw_families
    ):
        raise PreflightError("DABench family manifest counts are inconsistent")
    if expected_task_count is not None and len(raw_tasks) != expected_task_count:
        raise PreflightError("DABench family manifest task_count is unexpected")
    if expected_family_count is not None and len(raw_families) != expected_family_count:
        raise PreflightError("DABench family manifest family_count is unexpected")

    family_ids: set[str] = set()
    covered_tasks: list[str] = []
    for index, raw in enumerate(raw_families):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"families[{index}] must be an object")
        _exact_fields(
            raw,
            {"family_id", "raw_sha256", "public_train_sha256", "representative_task_id", "task_ids"},
            f"families[{index}]",
        )
        family_id = _safe_id(raw.get("family_id"), f"families[{index}].family_id")
        _digest(raw.get("raw_sha256"), f"families[{index}].raw_sha256")
        _digest(raw.get("public_train_sha256"), f"families[{index}].public_train_sha256")
        task_ids = raw.get("task_ids")
        if family_id in family_ids or not isinstance(task_ids, list) or not task_ids:
            raise PreflightError(f"families[{index}] identity/task_ids are invalid")
        normalized_tasks = [_safe_id(value, f"families[{index}].task_ids") for value in task_ids]
        if normalized_tasks != sorted(set(normalized_tasks)):
            raise PreflightError(f"families[{index}].task_ids must be unique and sorted")
        family_ids.add(family_id)
        covered_tasks.extend(normalized_tasks)

    task_fields = {
        "task_id",
        "family_id",
        "raw_filename",
        "raw_sha256",
        "public_train_sha256",
        "sample_submission_sha256",
        "private_answer_sha256",
        "vendor_contract_sha256",
        "vendor_contract_files",
    }
    task_entries: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"tasks[{index}] must be an object")
        _exact_fields(raw, task_fields, f"tasks[{index}]")
        task_id = _safe_id(raw.get("task_id"), f"tasks[{index}].task_id")
        family_id = _safe_id(raw.get("family_id"), f"tasks[{index}].family_id")
        raw_filename = _safe_id(raw.get("raw_filename"), f"tasks[{index}].raw_filename")
        for field in (
            "raw_sha256",
            "public_train_sha256",
            "sample_submission_sha256",
            "private_answer_sha256",
            "vendor_contract_sha256",
        ):
            _digest(raw.get(field), f"tasks[{index}].{field}")
        vendor_files = raw.get("vendor_contract_files")
        if family_id not in family_ids or task_id in task_entries or not isinstance(
            vendor_files, list
        ):
            raise PreflightError(f"tasks[{index}] identity/vendor contract is invalid")
        task_entries[task_id] = {**raw, "raw_filename": raw_filename}
    if sorted(covered_tasks) != sorted(task_entries):
        raise PreflightError("DABench families do not cover every task exactly once")

    selected: dict[str, Any] = {}
    task_to_family: dict[str, str] = {}
    for target in targets:
        task_id = target.task_id
        item = task_entries.get(task_id)
        if item is None:
            raise PreflightError(f"selected task {task_id!r} is missing from family manifest")
        if target.dataset_id != task_id:
            raise PreflightError("DABench family adaptation requires task-scoped identity")
        task_root = clean_root / task_id
        public_dir = task_root / "prepared" / "public"
        train = public_dir / "train.csv"
        sample = public_dir / "sample_submission.csv"
        private = task_root / "prepared" / "private" / "answer.csv"
        raw_path = task_root / "raw" / str(item["raw_filename"])
        if target.public_dir.resolve() != public_dir.resolve() or (
            target.sample_submission_path is None
            or target.sample_submission_path.resolve() != sample.resolve()
        ):
            raise PreflightError(f"selected task {task_id!r} does not use the clean release")
        task_to_family[task_id] = str(item["family_id"])
        selected[task_id] = {
            "family_id": item["family_id"],
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": item["raw_sha256"],
            "public_train_path": str(train.resolve()),
            "public_train_sha256": item["public_train_sha256"],
            "sample_submission_path": str(sample.resolve()),
            "sample_submission_sha256": item["sample_submission_sha256"],
            "private_answer_path": str(private.resolve()),
            "private_answer_sha256": item["private_answer_sha256"],
            "vendor_contract_sha256": item["vendor_contract_sha256"],
            "vendor_contract_files": item["vendor_contract_files"],
        }
    return task_to_family, {
        "path": str(manifest),
        "sha256": sha256(payload),
        "schema_version": DATASET_FAMILY_MANIFEST_SCHEMA_VERSION,
        "source_root": document["source_root"],
        "clean_root": str(clean_root),
        "declared_task_count": len(raw_tasks),
        "declared_family_count": len(raw_families),
        "selected_task_count": len(selected),
        "selected_family_count": len(set(task_to_family.values())),
        "vendor_root": str(vendor_root.resolve()) if vendor_root is not None else None,
        "tasks": selected,
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
    canonical_summaries: Mapping[str, Mapping[str, Any]],
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
            document, canonical_payload = _strict_json(source, f"canonical L1 for {family_id}")
            resolved_document = {**document, "dataset_id": target.task_id, "task_id": target.task_id}
            resolved_payload = (
                json.dumps(resolved_document, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            destination = condition_dir / f"{target.task_id}.json"
            try:
                destination.write_bytes(resolved_payload)
            except OSError as error:
                raise PreflightError(f"cannot write resolved L1 {destination}: {error}") from error
            try:
                canonical = load_l1_artifact(
                    skill.annotations_dir,
                    family_id,
                    expected_dataset_id=family_id,
                )
                resolved = load_l1_artifact(
                    condition_dir,
                    target.task_id,
                    expected_dataset_id=target.task_id,
                    expected_task_id=target.task_id,
                )
            except Exception as error:
                raise PreflightError(f"invalid resolved L1 {destination}: {error}") from error
            canonical_render = canonical.render_markdown(heading_level=3)
            resolved_render = resolved.render_markdown(heading_level=3)
            expected = canonical_summaries[condition_id]["canonical_files"][family_id]
            if resolved_render != canonical_render or sha256(canonical_payload) != expected["sha256"]:
                raise PreflightError(
                    f"resolved L1 changed canonical content for task {target.task_id!r}"
                )
            files[target.task_id] = {
                "family_id": family_id,
                "canonical_sha256": expected["sha256"],
                "resolved_sha256": sha256(resolved_payload),
                "rendered_l1_sha256": sha256(resolved_render.encode("utf-8")),
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


def _verify_family_execution_evaluation_hashes(
    summary: Mapping[str, Any] | None,
    *,
    phase: str,
    condition_id: str,
) -> Mapping[str, Any] | None:
    if summary is None:
        return None
    tasks = summary.get("tasks")
    if not isinstance(tasks, Mapping):
        raise PreflightError("DABench family summary has no selected tasks")
    vendor_root = Path(str(summary["vendor_root"])) if summary.get("vendor_root") else None
    attested: dict[str, Any] = {}
    for task_id in sorted(tasks):
        expected = tasks[task_id]
        if not isinstance(expected, Mapping):
            raise PreflightError(f"DABench scoring entry is invalid for {task_id}")
        actual = {
            name: _attest_file(
                Path(str(expected[path_field])),
                str(expected[hash_field]),
                f"{phase} {condition_id}: DABench {name} for {task_id}",
            )
            for name, path_field, hash_field in (
                ("raw", "raw_path", "raw_sha256"),
                ("public_train", "public_train_path", "public_train_sha256"),
                ("sample_submission", "sample_submission_path", "sample_submission_sha256"),
                ("private_answer", "private_answer_path", "private_answer_sha256"),
            )
        }
        if vendor_root is not None:
            vendor_hash, vendor_files = _vendor_contract(vendor_root / task_id, task_id)
            if vendor_hash != expected.get("vendor_contract_sha256") or vendor_files != expected.get(
                "vendor_contract_files"
            ):
                raise PreflightError(
                    f"{phase} {condition_id}: DABench vendor contract changed for {task_id}"
                )
            actual["vendor_contract"] = vendor_hash
        attested[task_id] = actual
    bundle = {"schema_version": "dabench_scoring_inputs_bundle_v1", "tasks": attested}
    return {
        "schema_version": "dabench_scoring_inputs_attestation_v1",
        "task_count": len(attested),
        "sha256": _canonical_sha256(bundle),
        "tasks": attested,
    }


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
        frozen = profile.frozen_input_sha256
        paths = {
            "dataset_family_manifest": family_manifest,
            "annotation_publication_manifest": (
                annotations_root / "annotation_publication_manifest.json"
            ),
            "skills_manifest": skills_manifest,
            "dataset_semantics_skill": (
                skills_manifest.parent / "perceive-dataset-semantics" / "SKILL.md"
            ),
        }
        try:
            for name, path in paths.items():
                actual = sha256(read_bytes(path, name.replace("_", " ")))
                if actual != frozen.get(name):
                    raise PreflightError(f"frozen DABench input hash mismatch: {name}")
            annotation_tree = tree_sha256(annotations_root)
            if annotation_tree != frozen.get("annotations_tree"):
                raise PreflightError("frozen DABench annotation tree hash mismatch")
            publication, _ = _strict_json(
                paths["annotation_publication_manifest"],
                "annotation publication manifest",
            )
            if publication.get("annotation_input_index_sha256") != frozen.get(
                "annotation_input_index"
            ):
                raise PreflightError("DABench annotation evidence hash mismatch")
        except Exception as error:
            raise AdapterPreflightError(
                f"DABench annotation publication verification failed: {error}"
            ) from error
        return {
            "benchmark": profile.benchmark,
            "data_root": str(data_root),
            "dataset_family_manifest": str(family_manifest),
            "annotation_publication_verified": True,
            "annotation_tree_sha256": annotation_tree,
            "annotation_publication": {
                key: value for key, value in publication.items() if key != "artifacts"
            },
        }

    def prepare_family_inputs(
        self,
        *,
        manifest_path: Path,
        targets: Sequence[Any],
        data_root: Path,
        vendor_root: Path,
        perception_skills: Sequence[PerceptionSkillCondition],
    ) -> DABenchFamilyInputs:
        mapping, summary = _load_dataset_family_manifest(
            manifest_path,
            targets,
            expected_clean_root=data_root,
            vendor_root=vendor_root,
            expected_task_count=EXPECTED_TASK_COUNT,
            expected_family_count=EXPECTED_FAMILY_COUNT,
        )
        canonical = {
            skill.perception_skill_id: _preflight_family_l1(
                skill.annotations_dir,
                targets,
                mapping,
            )
            for skill in perception_skills
        }
        _verify_family_execution_evaluation_hashes(
            summary,
            phase="preflight",
            condition_id="all",
        )
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
            canonical_summaries=family_inputs.canonical_l1_summaries,
            run_root=run_root,
        )

    def verify_scoring_inputs(
        self,
        summary: Mapping[str, Any] | None,
        *,
        phase: str,
        condition_id: str,
    ) -> Mapping[str, Any] | None:
        return _verify_family_execution_evaluation_hashes(
            summary,
            phase=phase,
            condition_id=condition_id,
        )
