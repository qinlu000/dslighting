"""Bridge legacy file-submission payloads to the canonical task layout."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from dslighting.benchmark.grading.models import SubmissionArtifactContract
from dslighting.core.tasks.errors import TaskExecutionSpecError
from dslighting.core.tasks.models import ResolvedTaskLayout
from dslighting.core.tasks.resolver import TaskResolver


def resolve_file_submission_payload_layout(
    *,
    task_id: str,
    description: str,
    data_dir: Path,
    output_path: Path,
    registry_dir: str | Path | None,
    submission_contract: SubmissionArtifactContract | None,
    submission_context: Mapping[str, Any],
) -> ResolvedTaskLayout:
    """Normalize an old benchmark payload before prompt composition.

    The resolver supplies grading and source metadata.  Payload-owned runtime
    values (the selected output path and its submission contract) then replace
    resolver defaults so benchmark scheduling and grading keep their existing
    paths.
    """

    layout: ResolvedTaskLayout = TaskResolver().resolve(
        task_id=task_id,
        data=data_dir,
        registry_dir=registry_dir,
    )
    grading = layout.evaluation_contract.grading
    if grading is None:
        raise TaskExecutionSpecError(
            f"Task '{layout.task_id}' does not have an artifact submission contract."
        )

    effective_submission = (submission_contract or grading.submission).with_output_path(output_path)
    evaluation_contract = replace(
        layout.evaluation_contract,
        grading=replace(grading, submission=effective_submission),
    )
    resolved_submission_context = {
        **layout.submission_context,
        **submission_context,
        **effective_submission.to_payload(),
    }
    return replace(
        layout,
        description_text=description,
        agent_visible_dir=data_dir,
        output_path=output_path,
        sample_submission_path=effective_submission.sample_submission_path,
        submission_filename=effective_submission.submission_filename,
        submission_format=effective_submission.submission_format,
        submission_context=resolved_submission_context,
        evaluation_contract=evaluation_contract,
    )


__all__ = ["resolve_file_submission_payload_layout"]
