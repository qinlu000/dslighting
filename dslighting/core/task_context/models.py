"""Domain types and errors for solver-facing task-context policy.

This module intentionally has no L0 concept.  ``main`` is the unmodified main
task context; L1 and L2 are independent enrichments and L3 combines them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

TaskContextPolicyValue = Literal["main", "l1", "l2", "l3"]


class TaskContextPolicy(str, Enum):
    """The four supported solver-facing task-context policies."""

    MAIN = "main"
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"

    @classmethod
    def parse(cls, value: str | TaskContextPolicy) -> TaskContextPolicy:
        """Return a canonical policy or raise a domain-specific error."""

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise TaskContextPolicyError(
                f"Unsupported task-context policy {normalized!r}; "
                "expected one of: main, l1, l2, l3."
            ) from error

    @property
    def includes_l1(self) -> bool:
        return self in {self.L1, self.L3}

    @property
    def includes_l2(self) -> bool:
        return self in {self.L2, self.L3}


@dataclass(frozen=True)
class ArtifactProvenance:
    """Stable provenance for a loaded task-context artifact."""

    path: Path
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class TaskContextProvenance:
    """Reproducibility metadata for one compiled solver task context."""

    policy: TaskContextPolicy
    dataset_id: str
    main_data_report_sha256: str
    l1: ArtifactProvenance | None = None
    l2: ArtifactProvenance | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "dataset_id": self.dataset_id,
            "main_data_report_sha256": self.main_data_report_sha256,
            "l1": self.l1.as_dict() if self.l1 is not None else None,
            "l2": self.l2.as_dict() if self.l2 is not None else None,
        }


class TaskContextError(RuntimeError):
    """Base class for task-context domain failures."""


class TaskContextPolicyError(TaskContextError, ValueError):
    """Raised when an unsupported policy is requested."""


class TaskContextArtifactError(TaskContextError):
    """Base class for external L1/L2 artifact failures."""


class ArtifactNotFoundError(TaskContextArtifactError, FileNotFoundError):
    """Raised when a selected artifact is not a readable regular file."""


class ArtifactReadError(TaskContextArtifactError):
    """Raised when an artifact exists but cannot be read as UTF-8."""


class ArtifactValidationError(TaskContextArtifactError, ValueError):
    """Raised when artifact content violates its declared schema."""


class L1ArtifactIdentityError(ArtifactValidationError):
    """Raised when an L1 artifact identifies a different task or dataset."""


class L1PublicCoverageError(ArtifactValidationError):
    """Raised when an L1 reference is not covered by visible public data."""


class DescriptionReplacementError(TaskContextError, ValueError):
    """Base class for deterministic description-section replacement failures."""


class DataDescriptionSectionMissingError(DescriptionReplacementError):
    """Raised when a supported dataset-description section is absent."""


class DataDescriptionSectionDuplicateError(DescriptionReplacementError):
    """Raised when more than one supported dataset-description section exists."""


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactProvenance",
    "ArtifactReadError",
    "ArtifactValidationError",
    "DataDescriptionSectionDuplicateError",
    "DataDescriptionSectionMissingError",
    "DescriptionReplacementError",
    "L1ArtifactIdentityError",
    "L1PublicCoverageError",
    "TaskContextArtifactError",
    "TaskContextError",
    "TaskContextPolicy",
    "TaskContextPolicyError",
    "TaskContextPolicyValue",
    "TaskContextProvenance",
]
