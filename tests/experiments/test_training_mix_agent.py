from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from pathlib import Path

import pytest

import experiments.dare_bench.runner as dare_runner
import experiments.data_agent_rl.runner as data_agent_runner
import experiments.datamind_sql.runner as datamind_runner
from experiments.dare_bench.runner import RunSettings as DareRunSettings
from experiments.data_agent_rl.runner import RunSettings as DataAgentRunSettings
from experiments.datamind_sql.runner import RunSettings as DataMindRunSettings
from experiments.training_mix.agent import (
    TrainingMixAgent,
    _context_settings,
    _env_flag,
    reward_from_result,
)
from experiments.training_mix.protocol_reward import ProtocolScore


def test_context_settings_match_every_benchmark_runner() -> None:
    names = set(_context_settings())
    for settings_type in (DareRunSettings, DataAgentRunSettings, DataMindRunSettings):
        assert names <= {field.name for field in fields(settings_type)}


def test_training_mix_allows_one_protocol_feedback_retry() -> None:
    assert _context_settings()["max_feedback_retries"] == 1
    for settings_type in (DareRunSettings, DataAgentRunSettings, DataMindRunSettings):
        assert settings_type.__dataclass_fields__["max_feedback_retries"].default == 1


def test_perception_rl_requires_explicit_boolean_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINING_PERCEPTION_ENABLED", raising=False)
    assert _env_flag("TRAINING_PERCEPTION_ENABLED") is False

    monkeypatch.setenv("TRAINING_PERCEPTION_ENABLED", "1")
    assert _env_flag("TRAINING_PERCEPTION_ENABLED") is True

    monkeypatch.setenv("TRAINING_PERCEPTION_ENABLED", "maybe")
    with pytest.raises(ValueError, match="boolean flag"):
        _env_flag("TRAINING_PERCEPTION_ENABLED")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["dare_bench", "data_agent_rl", "datamind_sql"])
async def test_dispatch_routes_both_roles_through_the_same_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
) -> None:
    captured: dict[str, object] = {}

    async def capture_run(_tasks: object, settings: object) -> dict[str, object]:
        captured["settings"] = settings
        return {}

    monkeypatch.setenv("AGL_OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("TRAINING_PERCEPTION_ENABLED", "1")
    for name in (
        "DARE_MANIFEST",
        "DARE_DATABASES_DIR",
        "DARE_PYTHON",
        "DATA_AGENT_ROOT",
        "DATA_AGENT_CACHE",
        "GENERAL_PYTHON",
        "TRAINING_MIX_MANIFEST",
        "DATAMIND_ROOT",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    if source == "dare_bench":
        monkeypatch.setattr(dare_runner, "DareDataStore", lambda **_kwargs: object())
        monkeypatch.setattr(dare_runner, "load_tasks", lambda _path: [object()])
        monkeypatch.setattr(
            dare_runner,
            "select_tasks",
            lambda tasks, **_kwargs: tasks,
        )
        monkeypatch.setattr(dare_runner, "run_tasks", capture_run)
    elif source == "data_agent_rl":
        monkeypatch.setattr(
            data_agent_runner,
            "BucketDataStore",
            lambda **_kwargs: object(),
        )
        monkeypatch.setattr(data_agent_runner, "load_task", lambda *_args: object())
        monkeypatch.setattr(data_agent_runner, "run_tasks", capture_run)
    else:
        monkeypatch.setattr(
            datamind_runner,
            "DataMindSQLDataStore",
            lambda *_args: object(),
        )
        monkeypatch.setattr(datamind_runner, "load_tasks", lambda _path: [object()])
        monkeypatch.setattr(
            datamind_runner,
            "select_tasks",
            lambda tasks, **_kwargs: tasks,
        )
        monkeypatch.setattr(datamind_runner, "run_tasks", capture_run)

    await TrainingMixAgent()._dispatch(source, "task-1", tmp_path, "test-key")

    settings = captured["settings"]
    assert settings.perception_enabled is True
    assert settings.perception_llm is settings.llm


@pytest.mark.parametrize(
    ("source", "payload", "expected"),
    [
        ("dare_bench", {"score": 0.75, "grade": {}}, 0.75),
        ("data_agent_rl", {"reward": 1.0, "grade": {"method": "numeric"}}, 1.0),
        ("datamind_sql", {"reward": 0.0, "grade": {"method": "mismatch"}}, 0.0),
    ],
)
def test_reward_from_programmatic_result(
    tmp_path: Path, source: str, payload: dict, expected: float
) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert reward_from_result(source, path) == expected


def test_infrastructure_failure_is_not_converted_to_negative_reward(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"reward": 0.0, "grade": {"method": "error", "detail": "API down"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        reward_from_result("data_agent_rl", path)


@pytest.mark.asyncio
async def test_agent_deadline_posts_programmatic_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = TrainingMixAgent()
    posted: dict[str, object] = {}

    async def slow_task(*_args: object) -> float:
        await asyncio.sleep(1)
        return 1.0

    async def capture_reward(
        _event_url: str,
        _agl_key: str,
        source: str,
        task_id: str,
        reward: float,
        *,
        scorer: str = "programmatic",
        details: dict[str, object] | None = None,
    ) -> None:
        posted.update(
            source=source,
            task_id=task_id,
            reward=reward,
            scorer=scorer,
            details=details,
        )

    async def no_protocol_events(*_args: object) -> ProtocolScore:
        return ProtocolScore(
            value=0.0,
            valid_turns=0,
            total_turns=0,
            terminal_answer_valid=False,
        )

    monkeypatch.setenv("TRAINING_SOURCE", "dare_bench")
    monkeypatch.setenv("TRAINING_TASK_ID", "slow-task")
    monkeypatch.setenv("AGL_EVENT_URL", "http://unused")
    monkeypatch.setenv("TRAINING_AGENT_DEADLINE_SECONDS", "0.01")
    monkeypatch.setattr(agent, "_run_task", slow_task)
    monkeypatch.setattr(agent, "_load_protocol_score", no_protocol_events)
    monkeypatch.setattr(agent, "_post_reward", capture_reward)

    await agent.run()

    assert posted == {
        "source": "dare_bench",
        "task_id": "slow-task",
        "reward": 0.0,
        "scorer": "programmatic_timeout+strict_react_protocol",
        "details": {
            "task_reward": 0.0,
            "protocol_reward": 0.0,
            "protocol_weight": 0.1,
            "protocol_valid_turns": 0,
            "protocol_total_turns": 0,
            "protocol_terminal_answer_valid": False,
        },
    }
