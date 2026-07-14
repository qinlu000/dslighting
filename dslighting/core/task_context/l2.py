"""Strict, deterministic L2 guidance-artifact handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._artifact_io import normalize_markdown, read_utf8_artifact
from ._markdown import find_disallowed_context_content, scan_markdown_headings
from .models import ArtifactProvenance, ArtifactValidationError


@dataclass(frozen=True)
class L2MarkdownArtifact:
    """A non-empty UTF-8 L2 guidance document and its provenance."""

    markdown: str
    provenance: ArtifactProvenance

    @classmethod
    def load(cls, path: Path | str) -> L2MarkdownArtifact:
        text, provenance = read_utf8_artifact(Path(path).expanduser(), artifact_label="L2")
        markdown = normalize_markdown(text)
        if not markdown:
            raise ArtifactValidationError(f"L2 guidance artifact is empty: {path}")
        if "\x00" in markdown:
            raise ArtifactValidationError(f"L2 guidance artifact contains NUL: {path}")
        _validate_l2_section_boundary(markdown, path=provenance.path)
        return cls(markdown=markdown + "\n", provenance=provenance)

    def render_markdown(self) -> str:
        return f"## Level 2 Guidance\n\n{self.markdown}"


def load_l2_artifact(path: Path | str) -> L2MarkdownArtifact:
    """Convenience interface for fail-closed L2 Markdown loading."""

    return L2MarkdownArtifact.load(path)


_L2_FORBIDDEN_SECTION_TITLES = frozenset(
    {
        "data description",
        "dataset description",
        "grader facts",
        "grading facts",
        "i/o instructions",
        "input/output instructions",
        "level 0",
        "output contract",
        "private data",
        "submission artifact requirements",
        "submission format",
        "submission format requirements",
        "task answer",
        "critical i/o requirements",
    }
)


def _validate_l2_section_boundary(markdown: str, *, path: Path) -> None:
    """Keep L2 out of semantic, submission, and task-artifact I/O slots."""

    headings = scan_markdown_headings(markdown)
    structural = [heading.title for heading in headings if heading.level <= 2]
    if structural:
        raise ArtifactValidationError(
            f"L2 guidance may not declare level-one or level-two sections "
            f"{structural}; the runtime owns its section wrapper: {path}"
        )

    forbidden = [
        heading.title
        for heading in headings
        if any(
            heading.title.strip().lower() == title
            or heading.title.strip().lower().startswith(f"{title} ")
            or heading.title.strip().lower().startswith(f"{title}:")
            for title in _L2_FORBIDDEN_SECTION_TITLES
        )
    ]
    if forbidden:
        raise ArtifactValidationError(
            f"L2 guidance declares main-owned or semantic sections {forbidden}: {path}"
        )

    forbidden_prose = find_disallowed_context_content(markdown)
    if forbidden_prose:
        raise ArtifactValidationError(
            "L2 guidance contains main-owned or disallowed content " f"{forbidden_prose}: {path}"
        )


__all__ = ["L2MarkdownArtifact", "load_l2_artifact"]
