"""Run AgenticDataBench tasks through DSLighting without registry integration.

The adapter deliberately keeps the two systems' responsibilities separate:

* AgenticDataBench owns task selection, datasets, output names, and scoring.
* DSLighting owns the solving workflow, sandbox, workspace, and telemetry.

Only the benchmark question and domain data are exposed to the solving agent.
Gold files, evaluator functions, and benchmark skill labels are never included
in the DSLighting task payload.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dslighting.config import LLMConfig
from dslighting.core.types import TaskDefinition
from experiments.common import (
    RuntimeConfig,
    acquire_run_lock,
    create_runner,
    last_task_record,
    prepare_task_output,
    run_batch,
    write_json,
    write_jsonl,
)

PLOT_HELPER_RELATIVE_PATH = Path("testbed/da_agent/configs/scripts/image.py")
DEFAULT_TASKS_RELATIVE_PATH = Path("testbed/tasks/dev.jsonl")
DEFAULT_DATASETS_RELATIVE_PATH = Path("testbed/datasets")
SELECTED_TASKS_FILENAME = "agenticdatabench_tasks.jsonl"


def is_perception_sandbox_supported(
    *,
    backend: str,
    local_isolation: str,
    disable_network: bool,
) -> bool:
    """Return whether the requested sandbox can safely run Perception."""
    if not disable_network:
        return False
    if backend == "docker":
        return True
    return backend == "local" and local_isolation == "bubblewrap"


@dataclass(frozen=True)
class AgenticDataBenchTask:
    """Validated task fields plus the original record retained for scoring."""

    task_id: str
    question: str
    domain: str
    output_file_names: tuple[str, ...]
    has_plot_post_process: bool
    upstream_record: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AgenticDataBenchTask":
        task_id = str(payload.get("id") or "").strip()
        question = str(payload.get("question") or "").strip()
        domain = str(payload.get("domain") or "").strip()
        outputs = payload.get("output_file_name")
        if isinstance(outputs, str):
            output_file_names = (outputs.strip(),)
        elif isinstance(outputs, list):
            output_file_names = tuple(str(value).strip() for value in outputs)
        else:
            output_file_names = ()

        _validate_path_component(task_id, field="id")
        if not question:
            raise ValueError(f"Task {task_id!r} has no question")
        if not domain:
            raise ValueError(f"Task {task_id!r} has no domain")
        if not output_file_names or any(not value for value in output_file_names):
            raise ValueError(f"Task {task_id!r} has no valid output_file_name")
        for output_name in output_file_names:
            _validate_path_component(output_name, field=f"{task_id}.output_file_name")
        if len(set(output_file_names)) != len(output_file_names):
            raise ValueError(f"Task {task_id!r} declares duplicate output files")

        post_process = payload.get("post_process_func") or []
        if isinstance(post_process, str):
            post_process = [post_process]
        has_plot_post_process = any(
            str(value).strip().startswith("image_post_process(") for value in post_process
        )
        return cls(
            task_id=task_id,
            question=question,
            domain=domain,
            output_file_names=output_file_names,
            has_plot_post_process=has_plot_post_process,
            upstream_record=dict(payload),
        )


@dataclass(frozen=True)
class RunSettings:
    benchmark_root: Path
    dataset_root: Path
    output_dir: Path
    workspace_dir: Path
    staging_dir: Path
    workflow: str
    llm: LLMConfig
    max_steps: int = 30
    max_history_chars: int = 48000
    keep_recent_turns: int = 14
    summary_trigger_turns: int = 18
    concurrency: int = 30
    timeout_seconds: int = 3600
    sandbox_backend: str = "local"
    docker_image: str | None = None
    memory_mb: int = 8192
    cpu_cores: float = 4.0
    pids_limit: int = 256
    local_isolation: str = "bubblewrap"
    sandbox_python: Path | None = None
    disable_network: bool = True
    perception_enabled: bool = False
    overwrite: bool = False
    retry_failed: bool = False


def _validate_path_component(value: str, *, field: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be one safe path component: {value!r}")


def load_tasks(path: Path) -> list[AgenticDataBenchTask]:
    """Load and validate the public AgenticDataBench task JSONL."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Task config does not exist: {path}")

    tasks: list[AgenticDataBenchTask] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            task = AgenticDataBenchTask.from_payload(payload)
            if task.task_id in seen:
                raise ValueError(f"Duplicate task id in {path}: {task.task_id}")
            seen.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"Task config is empty: {path}")
    return tasks


def write_selected_task_config(
    tasks: Sequence[AgenticDataBenchTask],
    destination: Path,
) -> Path:
    """Write exact upstream records for only the tasks evaluated in this run.

    This file lives beside the experiment results and is never staged into the
    solving agent's workspace. It may therefore retain evaluator-only fields
    such as gold file names and evaluation functions for the official scorer.
    """

    destination = Path(destination).expanduser().resolve()
    task_ids = [task.task_id for task in tasks]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Selected tasks must be non-empty and unique")
    for task in tasks:
        if str(task.upstream_record.get("id") or "") != task.task_id:
            raise ValueError(f"Task {task.task_id!r} has no matching upstream record")

    return write_jsonl(destination, (task.upstream_record for task in tasks))


def select_tasks(
    tasks: Sequence[AgenticDataBenchTask],
    *,
    task_ids: Sequence[str] = (),
    index_expression: str | None = None,
    select_all: bool = False,
) -> list[AgenticDataBenchTask]:
    """Select tasks by exact id, upstream-compatible indices, or explicit all."""

    selectors = bool(task_ids) + bool(index_expression) + bool(select_all)
    if selectors != 1:
        raise ValueError("Choose exactly one of task_ids, index_expression, or select_all")

    if task_ids:
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {missing}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task ids must be unique")
        return [by_id[task_id] for task_id in task_ids]

    if select_all:
        return list(tasks)

    expression = str(index_expression or "").strip()
    if not expression:
        raise ValueError("Index expression must be non-empty")
    try:
        if "-" in expression:
            start_text, end_text = expression.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end <= start or end > len(tasks):
                raise ValueError
            return list(tasks[start:end])
        indices = [int(value.strip()) for value in expression.split(",")]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError
        if any(index < 0 or index >= len(tasks) for index in indices):
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Invalid index expression {expression!r}; use '0-5' or '0,2,4'") from exc
    return [tasks[index] for index in indices]


def required_output_names(task: AgenticDataBenchTask) -> tuple[str, ...]:
    """Return evaluator-visible outputs, including hidden plot sidecars."""

    required = list(task.output_file_names)
    if task.has_plot_post_process:
        for output_name in task.output_file_names:
            path = Path(output_name)
            if path.suffix.lower() != ".png":
                continue
            for suffix in (".json", ".npy"):
                sidecar = f"{path.stem}{suffix}"
                if sidecar not in required:
                    required.append(sidecar)
    return tuple(required)


def _plot_instructions(task: AgenticDataBenchTask) -> str:
    image_names = [name for name in task.output_file_names if Path(name).suffix.lower() == ".png"]
    if not task.has_plot_post_process or not image_names:
        return ""
    rendered = ", ".join(f"`{name}`" for name in image_names)
    return f"""

## Plot instrumentation (required)

The benchmark evaluates both the rendered PNG and structured matplotlib data.
For every required plot, call the bundled helper after plotting is complete and
before closing the figure:

```python
from image import Plotprocess
fig = plt.gcf()
Plotprocess.plot_process(fig, "{image_names[0]}")
```

Use only these image file names: {rendered}. The helper writes the PNG plus the
matching `.json` and `.npy` files required by the evaluator.
""".strip()


def _io_instructions(task: AgenticDataBenchTask, output_dir: Path) -> str:
    required = required_output_names(task)
    rendered = ", ".join(f"`{name}`" for name in required)
    return (
        "All benchmark input files are in the current working directory.\n"
        f"Create the directory `{output_dir.name}` in the current working directory.\n"
        f"Save every final output inside that directory. Required files: {rendered}.\n"
        "Do not rename the files and do not place final outputs outside that directory.\n"
        "Each execution Action may run in a fresh process: do not rely on in-memory "
        "state carrying across Actions. Files written to the workspace persist.\n"
        "Before finishing, verify that every required file exists."
    )


def build_task_definition(
    task: AgenticDataBenchTask,
    *,
    agent_visible_dir: Path,
    output_dir: Path,
) -> TaskDefinition:
    """Build a direct execution spec that bypasses DSLighting's registry resolver."""

    agent_visible_dir = Path(agent_visible_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    required = required_output_names(task)
    description = task.question
    plot_instructions = _plot_instructions(task)
    if plot_instructions:
        description = f"{description}\n\n{plot_instructions}"

    execution_spec = {
        "task_id": task.task_id,
        # The upstream evaluator owns grading; `datasci` prevents registry grading.
        "task_type": "datasci",
        "description_text": description,
        "io_instructions": _io_instructions(task, output_dir),
        "agent_visible_dir": str(agent_visible_dir),
        "output_path": str(output_dir),
        "submission_artifact_contract": {
            "output_submission_path": str(output_dir),
            "submission_filename": output_dir.name,
            "submission_format": "",
            "root_kind": "directory",
            "root_basename": output_dir.name,
            "entries": [
                {
                    "relative_path": name,
                    "format": Path(name).suffix.lower().lstrip("."),
                    "description": "AgenticDataBench required output",
                }
                for name in required
            ],
        },
    }
    return TaskDefinition(
        task_id=task.task_id,
        task_type="datasci",
        payload={"execution_spec": execution_spec},
    )


def resolve_domain_dir(dataset_root: Path, domain: str) -> Path:
    """Resolve a nested upstream domain without allowing traversal."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    candidate = (dataset_root / domain).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Domain escapes dataset root: {domain!r}") from exc
    if not candidate.is_dir():
        raise ValueError(f"Dataset domain directory does not exist: {candidate}")
    return candidate


def prepare_agent_visible_dir(
    task: AgenticDataBenchTask,
    *,
    dataset_root: Path,
    staging_root: Path,
    plot_helper: Path,
) -> Path:
    """Create a persistent, lightweight view of the official domain directory."""

    source_dir = resolve_domain_dir(dataset_root, task.domain)
    staging_root = Path(staging_root).expanduser().resolve()
    stage_dir = staging_root / task.task_id
    stage_dir.mkdir(parents=True, exist_ok=True)

    for source_item in source_dir.iterdir():
        destination = stage_dir / source_item.name
        if destination.exists() or destination.is_symlink():
            continue
        os.symlink(
            source_item.resolve(),
            destination,
            target_is_directory=source_item.is_dir(),
        )

    if task.has_plot_post_process:
        plot_helper = Path(plot_helper).expanduser().resolve()
        if not plot_helper.is_file():
            raise ValueError(f"AgenticDataBench plot helper does not exist: {plot_helper}")
        shutil.copy2(plot_helper, stage_dir / "image.py")
    return stage_dir


def _load_messages(record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    workspace_dir = Path(str(record.get("workspace_dir") or ""))
    if not workspace_dir.name:
        return []
    artifacts = workspace_dir / "artifacts"
    candidates = (
        (artifacts / "messages.json", False),
        (artifacts / "minisweagent_trajectory.json", True),
    )
    for path, nested in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if nested and isinstance(payload, Mapping):
            payload = payload.get("messages")
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return payload
    return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported result metadata type: {type(value).__name__}")


def write_upstream_result(
    task: AgenticDataBenchTask,
    *,
    output_dir: Path,
    result: Any,
    cost: float,
    usage: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write the metadata shim required by AgenticDataBench's evaluator."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = required_output_names(task)
    missing = [name for name in required if not (output_dir / name).is_file()]
    error_result = isinstance(result, str) and result.startswith("[ERROR]")
    finished = not missing and not error_result
    messages = _load_messages(record)
    steps = sum(message.get("role") == "assistant" for message in messages)
    added_files = [
        str(path.resolve())
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and "dabench" not in path.relative_to(output_dir).parts
    ]

    payload = {
        "finished": finished,
        "steps": steps,
        "result": _json_safe(result),
        "result_files": {"added_files": added_files, "changed_files": []},
        "task": task.question,
        "trajectory": messages,
        "missing_files": missing,
        "dslighting": {
            "cost": float(cost),
            "usage": _json_safe(dict(usage or {})),
            "workspace_dir": str(record.get("workspace_dir") or "") if record else "",
        },
    }
    result_path = output_dir / "dabench" / "result.json"
    write_json(result_path, payload)
    return payload


async def run_tasks(
    tasks: Sequence[AgenticDataBenchTask],
    settings: RunSettings,
) -> dict[str, Any]:
    """Run selected tasks with bounded concurrency and write a summary."""

    if not tasks:
        raise ValueError("At least one task must be selected")
    if (
        settings.max_steps <= 0
        or settings.max_history_chars <= 0
        or settings.keep_recent_turns <= 0
        or settings.timeout_seconds <= 0
    ):
        raise ValueError("max_steps, max_history_chars, and timeout_seconds must be positive")
    if settings.summary_trigger_turns < settings.keep_recent_turns:
        raise ValueError("summary_trigger_turns must be >= keep_recent_turns")
    if settings.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if settings.memory_mb <= 0 or settings.cpu_cores <= 0 or settings.pids_limit <= 0:
        raise ValueError("memory_mb, cpu_cores, and pids_limit must be positive")
    if settings.sandbox_backend == "docker" and not str(settings.docker_image or "").strip():
        raise ValueError("Docker sandbox requires docker_image")
    if settings.perception_enabled and (
        settings.workflow != "react"
        or not is_perception_sandbox_supported(
            backend=settings.sandbox_backend,
            local_isolation=settings.local_isolation,
            disable_network=settings.disable_network,
        )
    ):
        raise ValueError(
            "Perception requires ReAct with either an offline Docker sandbox "
            "or an offline local Bubblewrap sandbox"
        )
    output_root = settings.output_dir.expanduser().resolve()
    workspace_root = settings.workspace_dir.expanduser().resolve()
    staging_root = settings.staging_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_run_lock(output_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    selected_tasks_file = write_selected_task_config(
        tasks,
        output_root / SELECTED_TASKS_FILENAME,
    )

    sandbox = {
        "backend": settings.sandbox_backend,
        "timeout": settings.timeout_seconds,
        "memory_mb": settings.memory_mb,
        "cpu_cores": settings.cpu_cores,
        "pids_limit": settings.pids_limit,
    }
    if settings.sandbox_backend in {"local", "docker"}:
        sandbox.update(
            {
                "environment_policy": "allowlist",
                "network_policy": "disabled" if settings.disable_network else "inherit",
            }
        )
    if settings.sandbox_backend == "local":
        sandbox["local_isolation"] = settings.local_isolation
        if settings.sandbox_python is not None:
            sandbox["python_executable"] = str(settings.sandbox_python.expanduser().absolute())
    elif settings.sandbox_backend == "docker":
        sandbox["docker_image"] = str(settings.docker_image).strip()

    agent_runtime: dict[str, Any] = {
        "max_steps": settings.max_steps,
        "perception_enabled": settings.perception_enabled,
        "context": {
            "max_history_chars": settings.max_history_chars,
            "keep_recent_turns": settings.keep_recent_turns,
            "summary_trigger_turns": settings.summary_trigger_turns,
        },
    }
    runner = create_runner(
        RuntimeConfig(
            workflow=settings.workflow,
            llm=settings.llm,
            workspace_dir=workspace_root,
            run_name=output_root.name,
            sandbox=sandbox,
            agent_runtime=agent_runtime,
            cpu_threads=max(1, int(settings.cpu_cores)),
        )
    )
    config = runner.config
    evaluate_task = runner.get_eval_function()
    plot_helper = settings.benchmark_root / PLOT_HELPER_RELATIVE_PATH

    async def run_one(task: AgenticDataBenchTask) -> dict[str, Any]:
        task_output = output_root / task.task_id
        existing = prepare_task_output(
            task_output,
            overwrite=settings.overwrite,
            retry_failed=settings.retry_failed,
            result_file="dabench/result.json",
            complete_field="finished",
            complete_value=True,
            create=False,
        )
        if existing is not None:
            print(f"skip task={task.task_id} reason=existing_result")
            return {"task_id": task.task_id, "status": "skipped"}

        print(f"run task={task.task_id} workflow={settings.workflow} model={settings.llm.model}")
        visible_dir = prepare_agent_visible_dir(
            task,
            dataset_root=settings.dataset_root,
            staging_root=staging_root,
            plot_helper=plot_helper,
        )
        definition = build_task_definition(
            task,
            agent_visible_dir=visible_dir,
            output_dir=task_output,
        )
        result, cost, usage = await evaluate_task(definition)
        record = last_task_record(runner.get_run_records(), task.task_id)
        upstream = write_upstream_result(
            task,
            output_dir=task_output,
            result=result,
            cost=cost,
            usage=usage,
            record=record,
        )
        status = "completed" if upstream["finished"] else "failed"
        task_summary = {
            "task_id": task.task_id,
            "status": status,
            "finished": upstream["finished"],
            "missing_files": upstream["missing_files"],
            "cost": cost,
            "workspace_dir": upstream["dslighting"]["workspace_dir"],
        }
        print(f"{status} task={task.task_id} cost={cost:.6f} missing={upstream['missing_files']}")
        return task_summary

    try:
        results = await run_batch(
            tasks,
            run_one,
            concurrency=settings.concurrency,
        )
    finally:
        lock_handle.close()

    summary = {
        "workflow": settings.workflow,
        "model": config.llm.model,
        "thinking": config.llm.thinking,
        "perception_enabled": settings.perception_enabled,
        "concurrency": settings.concurrency,
        "global_max_concurrency": config.llm.global_max_concurrency,
        "max_retries": config.llm.max_retries,
        "max_steps": settings.max_steps,
        "max_history_chars": settings.max_history_chars,
        "keep_recent_turns": settings.keep_recent_turns,
        "summary_trigger_turns": settings.summary_trigger_turns,
        "timeout_seconds": settings.timeout_seconds,
        "sandbox_backend": settings.sandbox_backend,
        "sandbox_python": str(config.sandbox.python_executable or ""),
        "task_config": str(selected_tasks_file),
        "task_count": len(tasks),
        "completed": sum(item.get("status") == "completed" for item in results),
        "failed": sum(item.get("status") == "failed" for item in results),
        "skipped": sum(item.get("status") == "skipped" for item in results),
        "total_cost": sum(float(item.get("cost") or 0.0) for item in results),
        "tasks": results,
    }
    write_json(output_root / "dslighting_run_summary.json", summary)
    return summary
