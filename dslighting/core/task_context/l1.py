"""Strict, deterministic L1 semantic-artifact handling for task contexts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ._artifact_io import read_utf8_artifact
from ._markdown import find_disallowed_context_content, replace_dataset_description
from .models import (
    ArtifactProvenance,
    ArtifactValidationError,
    L1ArtifactIdentityError,
    L1PublicCoverageError,
)
from .public_tree import PublicTreeAttestation, PublicTreeAttestor, iter_public_tree

L1_ARTIFACT_SCHEMA_VERSION: Literal["l1_semantic_map_v1"] = "l1_semantic_map_v1"

L1DataObjectKind = Literal[
    "table",
    "document",
    "image",
    "raster",
    "array",
    "geospatial",
    "archive",
    "bundle",
    "other",
]

_STRICT_MODEL = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_non_blank(value: str, *, label: str, single_line: bool = False) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL")
    if single_line and ("\n" in value or "\r" in value):
        raise ValueError(f"{label} must be a single line")
    return value


def _validate_public_relative_path(value: str, *, label: str) -> str:
    _validate_non_blank(value, label=label, single_line=True)
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError(f"{label} must be relative to the public root")
    if (
        not path.parts
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must not contain empty, '.' or '..' segments")
    return value


class L1DataObject(BaseModel):
    """One logical public-data object in an L1 semantic map."""

    model_config = _STRICT_MODEL

    name: str
    files: list[str] = Field(min_length=1)
    kind: L1DataObjectKind
    meaning: str
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return _validate_non_blank(value, label="data object name", single_line=True)

    @field_validator("meaning")
    @classmethod
    def _valid_meaning(cls, value: str) -> str:
        return _validate_non_blank(value, label="data object meaning")

    @field_validator("notes")
    @classmethod
    def _valid_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_blank(value, label="data object notes")

    @field_validator("files")
    @classmethod
    def _valid_files(cls, values: list[str]) -> list[str]:
        validated = [
            _validate_public_relative_path(value, label="data object file") for value in values
        ]
        if len(validated) != len(set(validated)):
            raise ValueError("data object files must be unique")
        return validated


class L1Variable(BaseModel):
    """One named variable belonging to a declared L1 data object."""

    model_config = _STRICT_MODEL

    object: str
    file: str | None = None
    name: str
    meaning: str
    unit: str | None = None

    @field_validator("object", "name")
    @classmethod
    def _valid_identifier(cls, value: str) -> str:
        return _validate_non_blank(value, label="variable identifier", single_line=True)

    @field_validator("meaning")
    @classmethod
    def _valid_meaning(cls, value: str) -> str:
        return _validate_non_blank(value, label="variable meaning")

    @field_validator("unit")
    @classmethod
    def _valid_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_blank(value, label="variable unit", single_line=True)

    @field_validator("file")
    @classmethod
    def _valid_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_public_relative_path(value, label="variable file")


class L1Structure(BaseModel):
    """A structural fact relating declared L1 data objects."""

    model_config = _STRICT_MODEL

    description: str
    objects: list[str] = Field(default_factory=list)

    @field_validator("description")
    @classmethod
    def _valid_description(cls, value: str) -> str:
        return _validate_non_blank(value, label="structure description")

    @field_validator("objects")
    @classmethod
    def _valid_objects(cls, values: list[str]) -> list[str]:
        validated = [
            _validate_non_blank(value, label="structure object", single_line=True)
            for value in values
        ]
        if len(validated) != len(set(validated)):
            raise ValueError("structure object references must be unique")
        return validated


class _L1SemanticPayload(BaseModel):
    """Fail-closed JSON schema used before constructing the public artifact type."""

    model_config = _STRICT_MODEL

    schema_version: Literal["l1_semantic_map_v1"]
    dataset_id: str | None = None
    task_id: str | None = None
    data_objects: list[L1DataObject] = Field(min_length=1)
    variables: list[L1Variable] = Field(default_factory=list)
    structure: list[L1Structure] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    annotation_notes: list[str] = Field(default_factory=list)

    @field_validator("dataset_id", "task_id")
    @classmethod
    def _valid_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_blank(value, label="artifact identity", single_line=True)

    @field_validator("uncertainties", "annotation_notes")
    @classmethod
    def _valid_audit_lines(cls, values: list[str]) -> list[str]:
        return [_validate_non_blank(value, label="audit line") for value in values]

    @model_validator(mode="after")
    def _valid_references(self) -> _L1SemanticPayload:
        object_names = [item.name for item in self.data_objects]
        if len(object_names) != len(set(object_names)):
            raise ValueError("data object names must be unique")

        declared = set(object_names)
        variable_keys: set[tuple[str, str | None, str]] = set()
        for variable in self.variables:
            if variable.object not in declared:
                raise ValueError(
                    f"variable {variable.name!r} references undeclared object {variable.object!r}"
                )
            key = (variable.object, variable.file, variable.name)
            if key in variable_keys:
                raise ValueError(f"duplicate variable annotation: {key!r}")
            variable_keys.add(key)

        for item in self.structure:
            missing = [name for name in item.objects if name not in declared]
            if missing:
                raise ValueError(
                    f"structure references undeclared objects: {', '.join(sorted(missing))}"
                )
        return self


@dataclass(frozen=True)
class L1SemanticArtifact:
    """A strictly parsed external L1 artifact plus reproducibility metadata."""

    schema_version: str
    dataset_id: str | None
    task_id: str | None
    data_objects: tuple[L1DataObject, ...]
    variables: tuple[L1Variable, ...]
    structure: tuple[L1Structure, ...]
    uncertainties: tuple[str, ...]
    annotation_notes: tuple[str, ...]
    provenance: ArtifactProvenance

    @classmethod
    def load(
        cls,
        artifact_dir: Path | str,
        artifact_id: str,
        *,
        expected_dataset_id: str | None = None,
        expected_task_id: str | None = None,
    ) -> L1SemanticArtifact:
        """Load exactly ``{artifact_dir}/{artifact_id}.json`` without fallback."""

        normalized_artifact_id = _validate_artifact_id(artifact_id)
        path = Path(artifact_dir).expanduser() / f"{normalized_artifact_id}.json"
        text, provenance = read_utf8_artifact(path, artifact_label="L1")
        payload = _parse_l1_payload(text, path)

        resolved_task_id = expected_task_id or normalized_artifact_id
        if payload.task_id is not None and payload.task_id != resolved_task_id:
            raise L1ArtifactIdentityError(
                f"L1 artifact task_id {payload.task_id!r} does not match requested "
                f"task_id {resolved_task_id!r}: {path}"
            )
        if expected_dataset_id is not None and payload.dataset_id != expected_dataset_id:
            raise L1ArtifactIdentityError(
                f"L1 artifact dataset_id {payload.dataset_id!r} does not match expected "
                f"dataset_id {expected_dataset_id!r}: {path}"
            )

        return cls(
            schema_version=payload.schema_version,
            dataset_id=payload.dataset_id,
            task_id=payload.task_id,
            data_objects=tuple(payload.data_objects),
            variables=tuple(payload.variables),
            structure=tuple(payload.structure),
            uncertainties=tuple(payload.uncertainties),
            annotation_notes=tuple(payload.annotation_notes),
            provenance=provenance,
        )

    @classmethod
    def from_directory(
        cls,
        artifact_dir: Path | str,
        artifact_id: str,
        *,
        expected_dataset_id: str | None = None,
        expected_task_id: str | None = None,
    ) -> L1SemanticArtifact:
        """Named alias for callers that want the source shape to be explicit."""

        return cls.load(
            artifact_dir,
            artifact_id,
            expected_dataset_id=expected_dataset_id,
            expected_task_id=expected_task_id,
        )

    def render_markdown(self, *, heading_level: int = 3) -> str:
        """Render a canonical solver-facing semantic map.

        Audit-only ``uncertainties`` and ``annotation_notes`` are intentionally
        excluded.  Collections are sorted by stable semantic keys so equivalent
        JSON object ordering cannot change the prompt.
        """

        _validate_heading_level(heading_level)
        heading = "#" * heading_level
        subheading = "#" * (heading_level + 1)
        lines = [f"{heading} Level 1 Semantic Data Map", "", f"{subheading} Data Objects", ""]

        for data_object in sorted(self.data_objects, key=lambda item: item.name):
            meaning = _canonical_prose(data_object.meaning)
            lines.append(f"- {_inline_code(data_object.name)} ({data_object.kind}): {meaning}")
            files = ", ".join(_inline_code(value) for value in sorted(data_object.files))
            lines.append(f"  - Files: {files}")
            if data_object.notes is not None:
                lines.append(f"  - Notes: {_canonical_prose(data_object.notes)}")

        if self.variables:
            lines.extend(["", f"{subheading} Variables", ""])
            for variable in sorted(
                self.variables,
                key=lambda item: (item.object, item.file or "", item.name),
            ):
                label = f"{_inline_code(variable.object)}.{_inline_code(variable.name)}"
                lines.append(f"- {label}: {_canonical_prose(variable.meaning)}")
                if variable.file is not None:
                    lines.append(f"  - File: {_inline_code(variable.file)}")
                if variable.unit is not None:
                    lines.append(f"  - Unit: {_canonical_prose(variable.unit)}")

        if self.structure:
            lines.extend(["", f"{subheading} Structure", ""])
            for item in sorted(
                self.structure,
                key=lambda value: (
                    _canonical_prose(value.description),
                    tuple(sorted(value.objects)),
                ),
            ):
                lines.append(f"- {_canonical_prose(item.description)}")
                if item.objects:
                    objects = ", ".join(_inline_code(value) for value in sorted(item.objects))
                    lines.append(f"  - Objects: {objects}")

        return "\n".join(lines).rstrip() + "\n"

    def replace_dataset_description(self, description: str) -> str:
        """Replace exactly one supported dataset-description body with this L1."""

        return replace_dataset_description(description, self.render_markdown())


def load_l1_artifact(
    artifact_dir: Path | str,
    artifact_id: str,
    *,
    expected_dataset_id: str | None = None,
    expected_task_id: str | None = None,
) -> L1SemanticArtifact:
    """Convenience interface for strict external L1 loading."""

    return L1SemanticArtifact.load(
        artifact_dir,
        artifact_id,
        expected_dataset_id=expected_dataset_id,
        expected_task_id=expected_task_id,
    )


def validate_l1_public_coverage(
    artifact: L1SemanticArtifact,
    public_root: Path | str,
    *,
    excluded_paths: Iterable[Path | str] = (),
) -> PublicTreeAttestation:
    """Validate that every L1 file reference belongs to visible public input.

    ``excluded_paths`` should contain sample submissions and other output
    templates supplied by the builder.  Directory references are rejected when
    they contain an excluded artifact, preventing a broad directory reference
    from indirectly exposing an output template as input semantics.
    """

    root = Path(public_root).expanduser()
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise L1PublicCoverageError(f"Public root does not exist: {root}") from error
    if not resolved_root.is_dir():
        raise L1PublicCoverageError(f"Public root is not a directory: {resolved_root}")

    excluded = _resolve_excluded_paths(resolved_root, excluded_paths)
    object_names = [item.name for item in artifact.data_objects]
    if len(object_names) != len(set(object_names)):
        raise L1PublicCoverageError("L1 data object names must be unique")

    targets_by_object: dict[str, tuple[Path, ...]] = {}
    for data_object in artifact.data_objects:
        targets = tuple(
            _resolve_public_reference(
                resolved_root,
                file_path,
                excluded=excluded,
                label=f"data object {data_object.name!r}",
            )
            for file_path in data_object.files
        )
        targets_by_object[data_object.name] = targets

    covered_targets = tuple(target for targets in targets_by_object.values() for target in targets)
    covered_directories = tuple(target for target in covered_targets if target.is_dir())
    covered_exact = frozenset(covered_targets) - frozenset(covered_directories)
    uncovered: list[str] = []
    attestor = PublicTreeAttestor()
    for record in iter_public_tree(resolved_root, excluded_paths=excluded):
        attestor.add(record)
        if record.kind == "directory":
            continue
        resolved_candidate = record.resolved_path
        if resolved_candidate not in covered_exact and not any(
            resolved_candidate.is_relative_to(target) for target in covered_directories
        ):
            uncovered.append(record.relative_path)
    if uncovered:
        uncovered.sort()
        raise L1PublicCoverageError(
            "L1 artifact leaves visible public input files undescribed: " + ", ".join(uncovered)
        )

    for variable in artifact.variables:
        object_targets = targets_by_object.get(variable.object)
        if object_targets is None:
            raise L1PublicCoverageError(
                f"Variable {variable.name!r} references undeclared object {variable.object!r}"
            )
        if variable.file is None:
            continue
        variable_target = _resolve_public_reference(
            resolved_root,
            variable.file,
            excluded=excluded,
            label=f"variable {variable.name!r}",
        )
        if not any(
            variable_target == object_target
            or (object_target.is_dir() and variable_target.is_relative_to(object_target))
            for object_target in object_targets
        ):
            raise L1PublicCoverageError(
                f"Variable {variable.name!r} file {variable.file!r} does not belong to "
                f"data object {variable.object!r}"
            )

    declared = set(targets_by_object)
    for item in artifact.structure:
        missing = sorted(set(item.objects) - declared)
        if missing:
            raise L1PublicCoverageError(
                f"Structure references undeclared objects: {', '.join(missing)}"
            )
    return attestor.finish()


def _validate_l1_solver_visible_content(
    payload: _L1SemanticPayload,
    *,
    path: Path,
) -> None:
    """Reject treatment leakage from every L1 field rendered to the solver."""

    rendered_fields: list[tuple[str, str]] = []
    for data_object in payload.data_objects:
        rendered_fields.extend(
            (
                (f"data object {data_object.name!r} name", data_object.name),
                (f"data object {data_object.name!r} meaning", data_object.meaning),
            )
        )
        if data_object.notes is not None:
            rendered_fields.append((f"data object {data_object.name!r} notes", data_object.notes))
    for variable in payload.variables:
        rendered_fields.extend(
            (
                (f"variable {variable.name!r} name", variable.name),
                (f"variable {variable.name!r} meaning", variable.meaning),
            )
        )
        if variable.unit is not None:
            rendered_fields.append((f"variable {variable.name!r} unit", variable.unit))
    for index, structure in enumerate(payload.structure):
        rendered_fields.append((f"structure[{index}] description", structure.description))

    violations = [
        f"{field}: {', '.join(matches)}"
        for field, value in rendered_fields
        if (matches := find_disallowed_context_content(value))
    ]
    if violations:
        raise ArtifactValidationError(
            "L1 solver-visible semantics contain main-owned or disallowed content "
            f"{violations}: {path}"
        )


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonKeyError(key)
        payload[key] = value
    return payload


def _parse_l1_payload(text: str, path: Path) -> _L1SemanticPayload:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except _DuplicateJsonKeyError as error:
        raise ArtifactValidationError(
            f"L1 artifact contains duplicate JSON key {str(error)!r}: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(
            f"L1 artifact is not valid JSON at line {error.lineno}, column {error.colno}: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"L1 artifact root must be a JSON object: {path}")
    try:
        parsed = cast(_L1SemanticPayload, _L1SemanticPayload.model_validate(payload))
    except ValidationError as error:
        raise ArtifactValidationError(f"Invalid L1 semantic artifact {path}: {error}") from error
    _validate_l1_solver_visible_content(parsed, path=path)
    return parsed


def _validate_artifact_id(artifact_id: str) -> str:
    if not isinstance(artifact_id, str):
        raise ArtifactValidationError("artifact_id must be a string")
    if artifact_id != artifact_id.strip() or not artifact_id:
        raise ArtifactValidationError(
            "artifact_id must be non-empty without surrounding whitespace"
        )
    if (
        artifact_id in {".", ".."}
        or "/" in artifact_id
        or "\\" in artifact_id
        or "\x00" in artifact_id
    ):
        raise ArtifactValidationError("artifact_id is not safe for artifact path resolution")
    return artifact_id


def _resolve_excluded_paths(
    public_root: Path,
    excluded_paths: Iterable[Path | str],
) -> frozenset[Path]:
    resolved: set[Path] = set()
    for raw_path in excluded_paths:
        path = Path(raw_path).expanduser()
        candidate = path if path.is_absolute() else public_root / path
        resolved.add(candidate.resolve(strict=False))
    return frozenset(resolved)


def _resolve_public_reference(
    public_root: Path,
    relative_path: str,
    *,
    excluded: frozenset[Path],
    label: str,
) -> Path:
    try:
        _validate_public_relative_path(relative_path, label=f"{label} path")
    except ValueError as error:
        raise L1PublicCoverageError(str(error)) from error

    candidate = public_root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise L1PublicCoverageError(
            f"{label} references missing public path {relative_path!r}"
        ) from error
    try:
        resolved.relative_to(public_root)
    except ValueError as error:
        raise L1PublicCoverageError(
            f"{label} path escapes public root: {relative_path!r}"
        ) from error

    excluded_match = resolved in excluded
    if resolved.is_dir() and not excluded_match:
        excluded_match = any(path.is_relative_to(resolved) for path in excluded)
    if excluded_match:
        raise L1PublicCoverageError(
            f"{label} references excluded output/template path {relative_path!r}"
        )
    return resolved


def _validate_heading_level(heading_level: int) -> None:
    if not isinstance(heading_level, int) or isinstance(heading_level, bool):
        raise ArtifactValidationError("heading_level must be an integer")
    if heading_level < 1 or heading_level > 5:
        raise ArtifactValidationError("heading_level must be between 1 and 5")


def _canonical_prose(value: str) -> str:
    return " ".join(value.split())


def _inline_code(value: str) -> str:
    return "`" + value.replace("`", "\\`") + "`"


__all__ = [
    "L1_ARTIFACT_SCHEMA_VERSION",
    "L1DataObject",
    "L1DataObjectKind",
    "L1SemanticArtifact",
    "L1Structure",
    "L1Variable",
    "load_l1_artifact",
    "validate_l1_public_coverage",
]
