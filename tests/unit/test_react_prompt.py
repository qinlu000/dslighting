from dslighting.prompts.workflows.react import (
    create_perception_prompt,
    create_react_prompt,
)


def test_create_react_prompt_uses_structured_dslighting_format() -> None:
    expected = """Role: You are a Solving Agent that completes data tasks by working with local files and executing Python.
Instructions:
  Goal: Complete the user task, follow every stated constraint, and produce all required output artifacts.
  Think: Use <Think>...</Think> to assess the current evidence, decide the next necessary step, and avoid repeating completed work.
  Action: Use <Action>...</Action> to execute Python for inspecting or transforming data, training or evaluating models, and creating or verifying required output artifacts. The <Action> block must contain exactly one non-empty fenced ```python ... ``` block and no other content.
  State: Each Action runs in a fresh Python process. Python variables and imports do not persist between Actions, but files created in the current working directory do persist.
  Explore: Use <Explore>...</Explore> when you need information about the data. Ask the Perception Agent to inspect and explore the local task data, and clearly state what data-related information or findings it should return. The Exploration Request must be plain text.
  PerceptionResult: Use the returned PerceptionResult when it is consistent with the user task. Verify with <Action> only when it conflicts with the task or relies on unsupported assumptions.
  Answer: Use <Answer>...</Answer> only after the task is complete and every required output artifact has been created and verified. The <Answer> block must contain a concise plain-text completion message and no code block.
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
  Think: Use <Think>...</Think> to reason about the current task state and decide the next step.
  Action: Use <Action>...</Action> to inspect and analyze the local task data without creating or modifying files. The <Action> block must contain exactly one non-empty fenced Python code block and no other content. Each Action runs in a fresh Python process, so repeat all imports and recreate any required in-memory state.
  Report: Use <Report>...</Report> when you can answer the Solving Agent's Exploration Request. Return a concise answer and the relevant findings from the data.
  Protocol: Every reply MUST contain exactly two blocks in this order: a <Think>...</Think> block followed by exactly one <Action>...</Action> or <Report>...</Report> block. Do not output text outside these blocks, and always close every tag."""

    assert create_perception_prompt() == expected


def test_create_react_prompt_hides_perception_when_offline_sandbox_is_unavailable() -> None:
    expected = """Role: You are a Solving Agent that completes data tasks by working with local files and executing Python.
Instructions:
  Goal: Complete the user task, follow every stated constraint, and produce all required output artifacts.
  Think: Use <Think>...</Think> to assess the current evidence, decide the next necessary step, and avoid repeating completed work.
  Action: Use <Action>...</Action> to execute Python for inspecting or transforming data, training or evaluating models, and creating or verifying required output artifacts. The <Action> block must contain exactly one non-empty fenced ```python ... ``` block and no other content.
  State: Each Action runs in a fresh Python process. Python variables and imports do not persist between Actions, but files created in the current working directory do persist.
  Answer: Use <Answer>...</Answer> only after the task is complete and every required output artifact has been created and verified. The <Answer> block must contain a concise plain-text completion message and no code block.
  Protocol: Every reply MUST contain exactly two blocks in this order: a <Think>...</Think> block followed by exactly one of <Action>...</Action> or <Answer>...</Answer>. Do not output text outside these blocks, and always close every tag."""

    prompt = create_react_prompt(
        {
            "goal_and_data": "Inspect local data.",
            "io_instructions": "Return an answer.",
        },
        allow_explore=False,
    )

    assert prompt == expected
