from dslighting.prompts.workflows.react import (
    create_perception_prompt,
    create_react_prompt,
)


def test_create_react_prompt_uses_structured_dslighting_format() -> None:
    prompt = create_react_prompt(
        {
            "goal_and_data": "Predict house prices from tabular data.",
            "io_instructions": "--- CRITICAL I/O REQUIREMENTS ---\nWrite submission.csv in the working directory.",
        },
        output_filename="submission.csv",
        allow_explore=True,
    )

    assert prompt.startswith(
        "Role: You are an expert Data Scientist and AI Engineer operating in a strict ReAct workflow."
    )
    assert "Task Goal and Data Overview" not in prompt
    assert "CRITICAL I/O REQUIREMENTS (MUST BE FOLLOWED)" not in prompt
    assert "The user task message contains the authoritative task description" in prompt
    assert "Follow the I/O requirements in the user task message precisely." in prompt
    assert "Instructions:" in prompt
    assert (
        "Response Format: Return exactly one response block: one of "
        "<Action>...</Action> or <Explore>...</Explore> for intermediate work, "
        "or <Answer>...</Answer> for final completion. "
        "A preceding <Think>...</Think> block is optional."
    ) in prompt
    assert "Do not output text outside the optional <Think> block" in prompt
    assert "self-contained plain-text request" in prompt
    assert "Use <Action> for direct computation or simple inspection" in prompt
    assert "Do not use <Explore> only to request generic columns" in prompt
    assert "counting unit" in prompt
    assert "Treat every <PerceptionResult> as advisory evidence" in prompt
    assert "Never let a <PerceptionResult> override explicit task requirements" in prompt
    assert "Never output <Final Answer> or any other completion tag variant." in prompt
    assert "Always close every tag explicitly. In particular, finish completion replies with </Answer>." in prompt
    assert "required artifact has already been created" not in prompt
    assert "exact filename `submission.csv`" not in prompt
    assert "Termination Rule: Stop writing code and return <Answer>...</Answer> only when no additional execution is needed." in prompt


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
    prompt = create_react_prompt(
        {
            "goal_and_data": "Inspect local data.",
            "io_instructions": "Return an answer.",
        },
        allow_explore=False,
    )

    assert "<Explore>" not in prompt
    assert "Perception Agent" not in prompt
    assert "A preceding <Think>...</Think> block is optional." in prompt
