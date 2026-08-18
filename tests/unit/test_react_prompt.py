from dslighting.prompts.workflows.react import (
    create_perception_prompt,
    create_react_prompt,
)


def test_create_react_prompt_uses_structured_dslighting_format() -> None:
    expected = """Role: You are a Solving Agent responsible for completing data analysis tasks in a ReAct workflow.
Instructions:
  Goal: Solve the user task by analyzing the local task data and following all stated constraints.
  Think: Use <Think>...</Think> to reason about the current task state and decide the next step.
  Action: Use <Action>...</Action> for direct data analysis, inspection, or verification. It must contain exactly one non-empty fenced ```python ... ``` block. Each Action runs in a fresh Python process, so repeat all imports and recreate any required in-memory state.
  Explore: Use <Explore>...</Explore> when you need information about the data. Ask the Perception Agent to inspect and explore the local task data, and clearly state what data-related information or findings it should return. The Exploration Request must be plain text.
  PerceptionResult: Use the returned PerceptionResult when it is consistent with the user task. Verify with <Action> only when it conflicts with the task or relies on unsupported assumptions.
  Answer: Use <Answer>...</Answer> when the available evidence is sufficient to answer the user task. It must contain the final answer as plain text.
  Protocol: Every reply MUST contain exactly two blocks in this order: a <Think>...</Think> block followed by exactly one of <Action>...</Action>, <Explore>...</Explore>, or <Answer>...</Answer>. Do not output text outside these blocks, and always close every tag."""

    prompt = create_react_prompt(
        {
            "goal_and_data": "Predict house prices from tabular data.",
            "io_instructions": "--- CRITICAL I/O REQUIREMENTS ---\nWrite submission.csv in the working directory.",
        },
        output_filename="submission.csv",
        allow_explore=True,
    )

    assert prompt == expected


def test_create_perception_prompt_is_minimal_local_prompt() -> None:
    expected = """Role: You are a Perception Agent supporting a Solving Agent.
Instructions:
  Goal: Analyze the local task data to answer the Solving Agent's Exploration Request, using the Original Task as context.
  Constraints:
    - You may perform any operation needed to inspect and analyze the local task data, except creating or modifying files.
    - Each Action runs in a fresh Python process. Repeat all imports and recreate any required in-memory state in every Action.
    - Return a concise Report that directly answers the Solving Agent's Exploration Request and includes the relevant findings from the data.
  Response Format:
    - Every reply MUST contain exactly one of <Action>...</Action> or <Report>...</Report>, with no other text.
    - Use <Action> only for exactly one non-empty fenced ```python ... ``` block.
    - Use <Report> for a concise plain-text perception report with no code block.
    - You have at most four replies. Use as few Actions as possible and return <Report> as soon as the request is answered, no later than the fourth reply.
    - Do not output text outside the required tags, and always close every tag."""

    assert create_perception_prompt() == expected


def test_create_react_prompt_hides_perception_when_offline_sandbox_is_unavailable() -> None:
    expected = """Role: You are a Solving Agent responsible for completing data analysis tasks in a ReAct workflow.
Instructions:
  Goal: Solve the user task by analyzing the local task data and following all stated constraints.
  Think: Use <Think>...</Think> to reason about the current task state and decide the next step.
  Action: Use <Action>...</Action> for direct data analysis, inspection, or verification. It must contain exactly one non-empty fenced ```python ... ``` block. Each Action runs in a fresh Python process, so repeat all imports and recreate any required in-memory state.
  Answer: Use <Answer>...</Answer> when the available evidence is sufficient to answer the user task. It must contain the final answer as plain text.
  Protocol: Every reply MUST contain exactly two blocks in this order: a <Think>...</Think> block followed by exactly one of <Action>...</Action> or <Answer>...</Answer>. Do not output text outside these blocks, and always close every tag."""

    prompt = create_react_prompt(
        {
            "goal_and_data": "Inspect local data.",
            "io_instructions": "Return an answer.",
        },
        allow_explore=False,
    )

    assert prompt == expected
