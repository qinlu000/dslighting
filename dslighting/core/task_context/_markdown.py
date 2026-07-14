"""Markdown structure helpers shared by L1 replacement and L2 validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ._artifact_io import normalize_markdown
from .models import (
    DataDescriptionSectionDuplicateError,
    DataDescriptionSectionMissingError,
    DescriptionReplacementError,
)


@dataclass(frozen=True)
class MarkdownHeading:
    level: int
    title: str
    start: int
    end: int
    newline: str
    style: Literal["atx", "setext", "html"]


_HEADING_RE = re.compile(r"^[ ]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*$")
_SETEXT_HEADING_RE = re.compile(r"^[ ]{0,3}(=+|-+)[ \t]*$")
_HTML_STRUCTURAL_HEADING_RE = re.compile(r"<\s*h([12])(?:[\t ]|>|$)", re.IGNORECASE)
_FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_DATASET_DESCRIPTION_TITLES = frozenset({"Data Description", "Dataset Description"})


def scan_markdown_headings(text: str) -> list[MarkdownHeading]:
    """Scan structural Markdown headings while ignoring fenced code."""

    headings: list[MarkdownHeading] = []
    offset = 0
    fence_char: str | None = None
    fence_length = 0
    previous_text_line: tuple[str, int, str] | None = None

    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        newline = raw_line[len(content) :]
        fence = _FENCE_LINE_RE.match(content)
        if fence_char is not None:
            if fence is not None:
                marker = fence.group(1)
                remainder = fence.group(2)
                if (
                    marker[0] == fence_char
                    and len(marker) >= fence_length
                    and not remainder.strip(" \t")
                ):
                    fence_char = None
                    fence_length = 0
            previous_text_line = None
            offset += len(raw_line)
            continue

        if fence is not None:
            marker = fence.group(1)
            info_string = fence.group(2)
            # CommonMark forbids a backtick in the info string of a backtick
            # fence. Tilde fences have no corresponding restriction.
            if marker[0] == "~" or "`" not in info_string:
                fence_char = marker[0]
                fence_length = len(marker)
                previous_text_line = None
                offset += len(raw_line)
                continue

        setext_match = _SETEXT_HEADING_RE.match(content)
        if setext_match is not None and previous_text_line is not None:
            previous_content, previous_start, previous_newline = previous_text_line
            headings.append(
                MarkdownHeading(
                    level=1 if setext_match.group(1).startswith("=") else 2,
                    title=previous_content.strip(),
                    start=previous_start,
                    end=offset + len(raw_line),
                    newline=newline or previous_newline,
                    style="setext",
                )
            )
            previous_text_line = None
            offset += len(raw_line)
            continue

        match = _HEADING_RE.match(content)
        if match is not None:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
            headings.append(
                MarkdownHeading(
                    level=len(match.group(1)),
                    title=title,
                    start=offset,
                    end=offset + len(raw_line),
                    newline=newline,
                    style="atx",
                )
            )
            previous_text_line = None
        else:
            html_match = _HTML_STRUCTURAL_HEADING_RE.search(content)
            if html_match is not None:
                headings.append(
                    MarkdownHeading(
                        level=int(html_match.group(1)),
                        title=f"raw HTML h{html_match.group(1)}",
                        start=offset,
                        end=offset + len(raw_line),
                        newline=newline,
                        style="html",
                    )
                )
                previous_text_line = None
            elif content.strip():
                previous_text_line = (content, offset, newline)
            else:
                previous_text_line = None
        offset += len(raw_line)

    # ``splitlines(keepends=True)`` covers a final non-newline line. Empty
    # input needs no special case.
    return headings


_FORBIDDEN_PROSE_PATTERNS = (
    (
        "dataset description",
        re.compile(r"\b(?:data|dataset)\s+description\b", re.IGNORECASE),
    ),
    ("task answer", re.compile(r"\btask\s+answer\b", re.IGNORECASE)),
    (
        "private or hidden information",
        re.compile(r"\b(?:private|hidden)\b", re.IGNORECASE),
    ),
    (
        "grader or grading information",
        re.compile(r"\b(?:grader|grading)\b", re.IGNORECASE),
    ),
    (
        "Output Contract",
        re.compile(r"\boutput[\s_-]+contract\b", re.IGNORECASE),
    ),
    (
        "I/O instructions",
        re.compile(
            r"\b(?:i\s*/\s*o|input\s*/\s*output)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "submission instructions",
        re.compile(r"\bsubmission\b", re.IGNORECASE),
    ),
    (
        "output artifact instructions",
        re.compile(
            r"\boutput\s+(?:artifact|file|filename|format|path|requirements?|instructions?)\b",
            re.IGNORECASE,
        ),
    ),
)


def find_disallowed_context_content(text: str) -> list[str]:
    """Return treatment content that belongs to a main-owned context slot."""

    return [
        label for label, pattern in _FORBIDDEN_PROSE_PATTERNS if pattern.search(text) is not None
    ]


def replace_dataset_description(description: str, replacement_markdown: str) -> str:
    """Replace one canonical dataset-description body and preserve its heading.

    The body ends at the next Markdown heading of level one or two. Headings
    inside fenced code blocks are ignored. DABench uses ``Data Description``;
    MoSciBench uses ``Dataset Description``. Missing or ambiguous exact target
    sections fail closed instead of appending or guessing.
    """

    replacement = normalize_markdown(replacement_markdown)
    if not replacement:
        raise DescriptionReplacementError("Dataset-description replacement must not be empty")

    replacement_headings = scan_markdown_headings(replacement)
    if any(item.level <= 2 for item in replacement_headings):
        raise DescriptionReplacementError(
            "Dataset-description replacement may not contain level-one or level-two headings"
        )

    headings = scan_markdown_headings(description)
    named_sections = [
        item for item in headings if item.title in _DATASET_DESCRIPTION_TITLES
    ]
    if not named_sections:
        raise DataDescriptionSectionMissingError(
            "Expected exactly one level-two dataset-description section "
            "('Data Description' or 'Dataset Description'), found none"
        )
    if len(named_sections) != 1:
        raise DataDescriptionSectionDuplicateError(
            "Expected exactly one level-two dataset-description section "
            f"('Data Description' or 'Dataset Description'), found {len(named_sections)}"
        )

    target = named_sections[0]
    if target.level != 2 or target.style not in {"atx", "setext"}:
        raise DataDescriptionSectionMissingError(
            "The dataset-description section must be a Markdown level-two heading "
            "('Data Description' or 'Dataset Description')"
        )
    following = next(
        (item for item in headings if item.start > target.start and item.level <= target.level),
        None,
    )
    suffix_start = following.start if following is not None else len(description)
    newline = target.newline or ("\r\n" if "\r\n" in description else "\n")
    canonical_replacement = replacement.replace("\n", newline)

    prefix = description[: target.end]
    if not prefix.endswith(("\n", "\r")):
        prefix += newline
    middle = newline + canonical_replacement
    if following is not None:
        middle += newline + newline
    else:
        middle += newline
    return prefix + middle + description[suffix_start:]


__all__ = [
    "MarkdownHeading",
    "find_disallowed_context_content",
    "replace_dataset_description",
    "scan_markdown_headings",
]
