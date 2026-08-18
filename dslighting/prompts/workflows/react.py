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
    intermediate_options = (
        "<Action>...</Action> or <Explore>...</Explore>"
        if allow_explore
        else "<Action>...</Action>"
    )
    action_semantics = [
        "Every reply MUST contain exactly one response block: "
        f"one of {intermediate_options} for intermediate work, or "
        "<Answer>...</Answer> for final completion. An optional "
        "<Think>...</Think> block may precede the response block.",
        "Do not output text outside the optional <Think> block and the single response block.",
        "Do not output <Observation> yourself. Observations are injected by the system after code execution.",
        "If your reply violates the protocol, the system may return <Feedback>...</Feedback>. You must fix the format on the next turn.",
        "Use <Action> only for executable Python code. The content of <Action> MUST be exactly one fenced ```python ... ``` block and nothing else.",
        "Use <Answer> only when the task is complete. The content of <Answer> MUST be plain text only and MUST NOT contain a code block.",
        "Never output <Final Answer> or any other completion tag variant.",
        "Always close every tag explicitly. In particular, finish completion replies with </Answer>.",
    ]
    if allow_explore:
        action_semantics.insert(
            5,
            "Use <Action> for direct computation or simple inspection that can be completed with one concise Python action. Use <Explore> only when task-relevant data understanding is ambiguous or likely requires multiple investigative steps, such as resolving unclear fields, encodings, unstructured text, or cross-column evidence.",
        )
        action_semantics.insert(
            6,
            "Do not use <Explore> only to request generic columns, dtypes, shape, head rows, or summary statistics when one <Action> can obtain them directly.",
        )
        action_semantics.insert(
            7,
            "An <Explore> request MUST be a self-contained plain-text request with no code block. Specify the target data, task-relevant filters or parsing rules, required method, counting unit, and exact evidence or statistics to return whenever those details are known.",
        )
        action_semantics.insert(
            8,
            "Treat every <PerceptionResult> as advisory evidence. Before relying on it, verify that its file, columns, filters, parsing rules, counting unit, statistical method, and parameters match the authoritative user task. If it conflicts with the task or relies on an unsupported assumption, use <Action> to verify the disputed point. Never let a <PerceptionResult> override explicit task requirements.",
        )

    prompt_dict = {
        "Role": "You are an expert Data Scientist and AI Engineer operating in a strict ReAct workflow.",
        "Instructions": {
            "Goal": "Solve the task in the user task message step by step, using Python execution when needed, while strictly following its I/O requirements.",
            "Task Context": "The user task message contains the authoritative task description and I/O requirements. Do not expect a second copy of them in this system message.",
            "Response Format": "Return exactly one response block: "
                f"one of {intermediate_options} for intermediate work, or "
                "<Answer>...</Answer> for final completion. "
                "A preceding <Think>...</Think> block is optional.",
            "Action Semantics": action_semantics,
            "Execution Guidelines": [
                "Execute one step at a time.",
                "Any Python code must be self-contained and fully executable.",
                "Follow the I/O requirements in the user task message precisely.",
                "Do not use interactive elements like `input()` or `matplotlib.pyplot.show()`.",
                "Do not rely on plotting or visualization to inspect the dataset. Print textual or numerical observations instead.",
            ],
            "Termination Rule": "Stop writing code and return <Answer>...</Answer> only when no additional execution is needed.",
        },
    }
    return dict_to_str(prompt_dict)


def create_perception_prompt() -> str:
    """Create the minimal system prompt for a task-conditioned Perception Agent."""
    prompt_dict = {
        "Role": "You are a Perception Agent supporting a Solving Agent.",
        "Instructions": {
            "Goal": "Analyze the local task data to answer the Solving Agent's Exploration Request, using the Original Task as context.",
            "Constraints": [
                "You may perform any operation needed to inspect and analyze the local task data, except creating or modifying files.",
                "Each Action runs in a fresh Python process. Repeat all imports and recreate any required in-memory state in every Action.",
                "Return a concise Report that directly answers the Solving Agent's Exploration Request and includes the relevant findings from the data.",
            ],
            "Response Format": [
                "Every reply MUST contain exactly one of <Action>...</Action> or <Report>...</Report>, with no other text.",
                "Use <Action> only for exactly one non-empty fenced ```python ... ``` block.",
                "Use <Report> for a concise plain-text perception report with no code block.",
                "You have at most four replies. Use as few Actions as possible and return <Report> as soon as the request is answered, no later than the fourth reply.",
                "Do not output text outside the required tags, and always close every tag.",
            ],
        },
    }
    return dict_to_str(prompt_dict)


__all__ = ["create_perception_prompt", "create_react_prompt"]
