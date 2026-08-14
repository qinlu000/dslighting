"""Compile one resolved task layout into the solver execution contract."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from dslighting.core.tasks.models import ResolvedTaskLayout, TaskExecutionSpec


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


class TaskContextBuilder:
    """Build the normal solver context from canonical task metadata."""

    def __init__(
        self,
        perception_runtime: PerceptionRuntime | None,
    ) -> None:
        self._perception = perception_runtime

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

        data_report = self._build_main_data_report(layout)
        io_instructions = str(explicit_io_instructions or "").strip()
        if not io_instructions:
            io_instructions = self._build_main_io_instructions(layout, submission_contract)

        description_text = layout.description_text
        if data_report:
            description_text = f"{description_text}\n\n{data_report}"
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
