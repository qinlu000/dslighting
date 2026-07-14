"""Compile one resolved task layout into the solver execution contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from dslighting.config import TaskContextConfig

if TYPE_CHECKING:
    from dslighting.core.tasks.models import ResolvedTaskLayout, TaskExecutionSpec

from ._markdown import replace_dataset_description
from .l1 import load_l1_artifact, validate_l1_public_coverage
from .l2 import load_l2_artifact
from .models import (
    ArtifactNotFoundError,
    ArtifactProvenance,
    TaskContextPolicy,
    TaskContextProvenance,
)


class PerceptionRuntime(Protocol):
    """The narrow part of the main perception runtime required here."""

    def analyze_data(
        self,
        data_dir: Path,
        task_type: str,
        task_id: str,
        submission_context: dict[str, Any],
    ) -> str: ...

    def generate_io_instructions(
        self,
        output_filename: str,
        optimization_context: bool,
        submission_context: dict[str, Any],
    ) -> str: ...


@dataclass(frozen=True)
class _Treatment:
    description_text: str
    l2_markdown: str
    l1_provenance: ArtifactProvenance | None
    l2_provenance: ArtifactProvenance | None


class _DatasetContextCompiler:
    """Resolve only the experimental semantic and guidance treatments.

    Data perception and main-owned submission text are deliberately not
    implemented here.  This keeps the experimental policy independent from
    the stable main report and I/O paths.
    """

    def __init__(self, settings: TaskContextConfig) -> None:
        self._settings = settings
        self.policy = TaskContextPolicy.parse(settings.policy)

    @property
    def requires_canonical_layout(self) -> bool:
        return self.policy is not TaskContextPolicy.MAIN or self._settings.require_canonical_layout

    def compile(self, layout: ResolvedTaskLayout) -> _Treatment:
        description = layout.description_text
        l1_provenance = None
        l2_provenance = None
        l2_markdown = ""

        if self.policy.includes_l1:
            artifact_dir = self._required_path(
                self._settings.l1_artifact_dir,
                label="task_context.l1_artifact_dir",
            )
            artifact = load_l1_artifact(
                artifact_dir,
                layout.dataset_id,
                expected_dataset_id=layout.dataset_id,
                expected_task_id=layout.task_id,
            )
            validate_l1_public_coverage(
                artifact,
                layout.agent_visible_dir,
                excluded_paths=self._output_template_paths(
                    layout,
                    layout.agent_visible_dir,
                ),
            )
            description = replace_dataset_description(
                description,
                artifact.render_markdown(heading_level=3),
            )
            l1_provenance = artifact.provenance

        if self.policy.includes_l2:
            guidance_path = self._required_path(
                self._settings.l2_guidance_path,
                label="task_context.l2_guidance_path",
            )
            guidance = load_l2_artifact(guidance_path)
            l2_markdown = guidance.render_markdown()
            l2_provenance = guidance.provenance

        return _Treatment(
            description_text=description,
            l2_markdown=l2_markdown,
            l1_provenance=l1_provenance,
            l2_provenance=l2_provenance,
        )

    @staticmethod
    def _required_path(value: str | None, *, label: str) -> Path:
        normalized = str(value or "").strip()
        if not normalized:
            raise ArtifactNotFoundError(f"{label} is required by the selected task-context policy.")
        return Path(normalized).expanduser()

    @staticmethod
    def _output_template_paths(
        layout: ResolvedTaskLayout,
        public_data_dir: Path,
    ) -> tuple[Path, ...]:
        paths: list[Path] = [layout.output_path]
        if layout.sample_submission_path is not None:
            paths.append(layout.sample_submission_path)
        grading = layout.evaluation_contract.grading
        if grading is not None:
            paths.append(grading.submission.output_submission_path)
            if grading.submission.sample_submission_path is not None:
                paths.append(grading.submission.sample_submission_path)
            paths.extend(
                entry.sample_path
                for entry in grading.submission.entries
                if entry.sample_path is not None
            )
            paths.extend(
                public_data_dir / entry.relative_path
                for entry in grading.submission.entries
                if entry.relative_path
            )
        return tuple(dict.fromkeys(paths))


class TaskContextBuilder:
    """The single ``ResolvedTaskLayout -> TaskExecutionSpec`` seam.

    The facade owns composition, while the internal dataset compiler owns only
    the L1/L2 treatment.  The existing perception runtime remains the authority
    for the data report and I/O instructions.  Output Contract behavior stays
    on the existing workflow path, and the builder only carries the main-owned
    submission artifact contract into the execution spec.
    """

    def __init__(
        self,
        settings: TaskContextConfig,
        perception_runtime: PerceptionRuntime | None,
    ) -> None:
        self._compiler = _DatasetContextCompiler(settings)
        self._perception = perception_runtime

    @property
    def policy(self) -> TaskContextPolicy:
        return self._compiler.policy

    @property
    def requires_canonical_layout(self) -> bool:
        """Whether legacy payloads must resolve through the canonical layout seam."""

        return self._compiler.requires_canonical_layout

    def build(
        self,
        layout: ResolvedTaskLayout,
        *,
        explicit_io_instructions: str | None = None,
    ) -> TaskExecutionSpec:
        from dslighting.core.tasks.errors import TaskExecutionSpecError
        from dslighting.core.tasks.models import TaskExecutionSpec

        submission_contract = (
            layout.evaluation_contract.grading.submission
            if layout.evaluation_contract.grading is not None
            else None
        )
        if submission_contract is None:
            raise TaskExecutionSpecError(
                f"Task '{layout.task_id}' does not have an artifact submission contract."
            )

        treatment = self._compiler.compile(layout)
        data_report = self._build_main_data_report(layout)
        io_instructions = str(explicit_io_instructions or "").strip()
        if not io_instructions:
            io_instructions = self._build_main_io_instructions(layout, submission_contract)

        description_text = treatment.description_text
        if data_report:
            description_text = f"{description_text}\n\n{data_report}"
        if treatment.l2_markdown:
            description_text = f"{description_text}\n\n{treatment.l2_markdown}"

        provenance = TaskContextProvenance(
            policy=self.policy,
            dataset_id=layout.dataset_id,
            main_data_report_sha256=hashlib.sha256(data_report.encode("utf-8")).hexdigest(),
            l1=treatment.l1_provenance,
            l2=treatment.l2_provenance,
        )
        lower_is_better = (
            layout.evaluation_contract.evaluation_semantics.objective == "lower_is_better"
        )
        return TaskExecutionSpec(
            task_id=layout.task_id,
            task_type=layout.task_type,
            description_text=description_text,
            io_instructions=io_instructions,
            agent_visible_dir=layout.agent_visible_dir,
            output_path=layout.output_path,
            metric_name="score",
            lower_is_better=lower_is_better,
            source_id=layout.source_id,
            engine_id=layout.engine_id,
            submission_artifact_contract=submission_contract.with_output_path(layout.output_path),
            evaluation_contract_ref=layout.evaluation_contract_ref,
            task_context_provenance=provenance.as_dict(),
        )

    def _build_main_data_report(self, layout: ResolvedTaskLayout) -> str:
        if self._perception is None:
            return ""
        return self._perception.analyze_data(
            layout.agent_visible_dir,
            task_type=layout.task_type,
            task_id=layout.task_id,
            submission_context=layout.submission_context,
        )

    def _build_main_io_instructions(self, layout, submission_contract) -> str:
        if self._perception is not None:
            return self._perception.generate_io_instructions(
                layout.output_path.name,
                optimization_context=False,
                submission_context=layout.submission_context,
            )
        if submission_contract.root_kind == "directory":
            required_files = ", ".join(
                f"`{entry.relative_path}`"
                for entry in submission_contract.entries
                if entry.relative_path
            )
            return (
                "All input data files are located in the current working directory (./).\n"
                f"You MUST create the submission directory `{layout.output_path.name}` "
                "in the current working directory.\n"
                f"The directory must contain: {required_files}."
            )
        return (
            "All input data files are located in the current working directory (./).\n"
            f"You MUST save the final submission artifact to "
            f"`{layout.output_path.name}` in the current working directory."
        )


__all__ = ["TaskContextBuilder"]
