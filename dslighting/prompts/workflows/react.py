"""
ReAct Workflow Prompts

Provides prompt templates for the ReAct workflow.
"""

from typing import Dict, Optional

from dslighting.prompts.base import dict_to_str


def create_react_prompt(
    task_context: Dict,
    *,
    output_filename: Optional[str] = None,
    allow_explore: bool = False,
) -> str:
    """Create the static system prompt for the ReAct workflow.

    Task-specific content is rendered once in the user task message by
    ``ReActWorkflow._render_task_message``.  Keeping it out of this system
    prompt prevents the task description, I/O contract, and any attached
    dataset information from being sent twice on every turn.
    """
    _ = task_context
    _ = output_filename
    response_options = "<Action>...</Action>, <Explore>...</Explore>, or <Answer>...</Answer>"
    if not allow_explore:
        response_options = "<Action>...</Action> or <Answer>...</Answer>"

    instructions = {
        "Goal": "Solve the user task by analyzing the local task data and following all stated constraints.",
        "Think": "Use <Think>...</Think> to reason about the current task state and decide the next step.",
        "Action": "Use <Action>...</Action> for direct data analysis, inspection, or verification. It must contain exactly one non-empty fenced ```python ... ``` block. Each Action runs in a fresh Python process, so repeat all imports and recreate any required in-memory state.",
    }
    if allow_explore:
        instructions["Explore"] = (
            "Use <Explore>...</Explore> when you need information about the data. "
            "Ask the Perception Agent to inspect and explore the local task data, "
            "and clearly state what data-related information or findings it should "
            "return. The Exploration Request must be plain text."
        )
        instructions["PerceptionResult"] = (
            "Use the returned PerceptionResult when it is consistent with the user "
            "task. Verify with <Action> only when it conflicts with the task or relies "
            "on unsupported assumptions."
        )

    instructions["Answer"] = (
        "Use <Answer>...</Answer> when the available evidence is sufficient to answer "
        "the user task. It must contain the final answer as plain text."
    )
    instructions["Protocol"] = (
        "Every reply MUST contain exactly two blocks in this order: a "
        f"<Think>...</Think> block followed by exactly one of {response_options}. "
        "Do not output text outside these blocks, and always close every tag."
    )

    prompt_dict = {
        "Role": "You are a Solving Agent responsible for completing data analysis tasks in a ReAct workflow.",
        "Instructions": instructions,
    }
    return dict_to_str(prompt_dict)


def create_perception_prompt() -> str:
    """Create the minimal system prompt for a task-conditioned Perception Agent."""
    prompt_dict = {
        "Role": "You are a Perception Agent supporting a Solving Agent.",
        "Instructions": {
            "Goal": "Analyze the local task data to answer the Solving Agent's Exploration Request, using the Original Task as context.",
            "Think": "Use <Think>...</Think> to reason about the current task state and decide the next step.",
            "Action": "Use <Action>...</Action> to inspect and analyze the local task data without creating or modifying files. The <Action> block must contain exactly one non-empty fenced Python code block and no other content. Each Action runs in a fresh Python process, so repeat all imports and recreate any required in-memory state.",
            "Report": "Use <Report>...</Report> when you can answer the Solving Agent's Exploration Request. Return a concise answer and the relevant findings from the data.",
            "Protocol": "Every reply MUST contain exactly two blocks in this order: a <Think>...</Think> block followed by exactly one <Action>...</Action> or <Report>...</Report> block. Do not output text outside these blocks, and always close every tag.",
        },
    }
    return dict_to_str(prompt_dict)


__all__ = ["create_perception_prompt", "create_react_prompt"]
