"""Agent Lightning local-process bridge for one DSLighting training rollout."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from dslighting.config import LLMConfig
from experiments.training_mix.protocol_reward import (
    ProtocolScore,
    combine_rewards,
    model_protocol_turns_from_events,
    score_protocol_turns,
)

log = logging.getLogger(__name__)

DARE_GRADING_ERRORS = {
    "empty_id_columns_after_resolving_alignment",
    "missing_prediction",
    "missing_target_columns",
    "prediction_merge_failed",
    "prediction_missing_id_columns",
    "row_count_mismatch_after_merge",
    "unreadable_artifact",
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _context_settings() -> dict[str, Any]:
    return {
        "max_history_chars": 64000,
        "keep_recent_turns": 12,
        "summary_trigger_turns": 16,
        "recent_observation_window": 12,
        "keep_latest_feedback_only": False,
        "max_feedback_retries": 1,
    }


def reward_from_result(source: str, result_path: Path) -> float:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    grade = payload.get("grade") or {}
    if source == "dare_bench":
        error = grade.get("error")
        if error and error not in DARE_GRADING_ERRORS:
            raise RuntimeError(f"DARE rollout infrastructure failed: {error}")
        reward = payload.get("score", 0.0)
    elif source in {"data_agent_rl", "datamind_sql"}:
        if grade.get("method") == "error":
            raise RuntimeError(f"{source} rollout infrastructure failed: {grade.get('detail')}")
        reward = payload.get("reward", 0.0)
    else:
        raise ValueError(f"Unsupported training source: {source}")
    return max(0.0, min(1.0, float(reward)))


def _rollout_key(event_url: str) -> str:
    match = re.search(r"/rollouts?/([^/]+)", event_url)
    value = match.group(1) if match else "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96]


def _events_query_url(event_url: str) -> str:
    value, replacements = re.subn(
        r"/attempt/[^/]+/events$",
        "/events",
        event_url,
    )
    if replacements != 1:
        raise ValueError(f"Unrecognized AGL event URL: {event_url}")
    return value


class TrainingMixAgent:
    """Run one manifest task and report its deterministic reward to AGL."""

    async def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        source = os.environ["TRAINING_SOURCE"]
        task_id = os.environ["TRAINING_TASK_ID"]
        event_url = os.environ["AGL_EVENT_URL"]
        agl_key = os.environ.get("AGL_KEY", "")
        deadline = float(os.environ.get("TRAINING_AGENT_DEADLINE_SECONDS", "1650"))
        if deadline <= 0:
            raise ValueError("TRAINING_AGENT_DEADLINE_SECONDS must be positive")
        protocol_weight = float(os.environ.get("TRAINING_FORMAT_REWARD_WEIGHT", "0.1"))
        if not 0.0 <= protocol_weight <= 1.0:
            raise ValueError("TRAINING_FORMAT_REWARD_WEIGHT must be between 0 and 1")
        perception_enabled = _env_flag("TRAINING_PERCEPTION_ENABLED")
        protocol_scorer = (
            "role_aware_strict_react_protocol" if perception_enabled else "strict_react_protocol"
        )
        scorer = f"programmatic+{protocol_scorer}"
        try:
            task_reward = await asyncio.wait_for(
                self._run_task(source, task_id, event_url, agl_key),
                timeout=deadline,
            )
        except asyncio.TimeoutError:
            # Finish before Agent Lightning's outer rollout timeout so slow
            # model-generated code receives a deterministic zero instead of
            # disappearing from the GRPO group without a reward.
            task_reward = 0.0
            scorer = f"programmatic_timeout+{protocol_scorer}"
            log.warning(
                "rollout deadline reached source=%s task=%s deadline=%.1fs",
                source,
                task_id,
                deadline,
            )
        protocol_score = await self._load_protocol_score(
            event_url,
            agl_key,
            perception_enabled,
        )
        reward = combine_rewards(
            task_reward,
            protocol_score.value,
            protocol_weight=protocol_weight,
        )
        reward_details = {
            "task_reward": task_reward,
            "protocol_reward": protocol_score.value,
            "protocol_weight": protocol_weight,
            "protocol_valid_turns": protocol_score.valid_turns,
            "protocol_total_turns": protocol_score.total_turns,
            "protocol_terminal_answer_valid": protocol_score.terminal_answer_valid,
        }
        if perception_enabled:
            reward_details.update(
                {
                    "solver_protocol_valid_turns": protocol_score.solver_valid_turns,
                    "solver_protocol_total_turns": protocol_score.solver_total_turns,
                    "perception_protocol_valid_turns": (protocol_score.perception_valid_turns),
                    "perception_protocol_total_turns": (protocol_score.perception_total_turns),
                }
            )
        await self._post_reward(
            event_url,
            agl_key,
            source,
            task_id,
            reward,
            scorer=scorer,
            details=reward_details,
        )
        log.info(
            "rollout source=%s task=%s reward=%.6f task_reward=%.6f "
            "protocol_reward=%.6f protocol_turns=%d/%d terminal_answer=%s "
            "scorer=%s",
            source,
            task_id,
            reward,
            task_reward,
            protocol_score.value,
            protocol_score.valid_turns,
            protocol_score.total_turns,
            protocol_score.terminal_answer_valid,
            scorer,
        )

    @staticmethod
    async def _load_protocol_score(
        event_url: str,
        agl_key: str,
        perception_enabled: bool,
    ) -> ProtocolScore:
        headers = {"Authorization": f"Bearer {agl_key}"}
        query_url = _events_query_url(event_url)
        async with httpx.AsyncClient(timeout=15.0) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.get(
                        query_url,
                        params={"event_type": "model_request"},
                        headers=headers,
                    )
                    response.raise_for_status()
                    events = response.json()
                    if not isinstance(events, list):
                        raise RuntimeError("AGL events response must be a list")
                    return score_protocol_turns(
                        model_protocol_turns_from_events(events),
                        perception_enabled=perception_enabled,
                    )
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(attempt)
        raise AssertionError("unreachable")

    async def _run_task(
        self,
        source: str,
        task_id: str,
        event_url: str,
        agl_key: str,
    ) -> float:
        scratch_root = Path(os.environ["TRAINING_SCRATCH_ROOT"]).expanduser().resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)
        prefix = f"rollout-{_rollout_key(event_url)}-"
        with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root) as temporary:
            root = Path(temporary)
            summary = await self._dispatch(source, task_id, root, agl_key)
            result_path = Path(summary["tasks"][0]["result"])
            return reward_from_result(source, result_path)

    async def _dispatch(
        self,
        source: str,
        task_id: str,
        root: Path,
        agl_key: str,
    ) -> dict[str, Any]:
        max_steps = int(os.environ.get("TRAINING_MAX_STEPS", "10"))
        timeout = int(os.environ.get("TRAINING_TIMEOUT_SECONDS", "1800"))
        threads = int(os.environ.get("TRAINING_COMPUTE_THREADS", "2"))
        context_settings = _context_settings()
        llm = LLMConfig(
            model=os.environ.get("TRAINING_MODEL_NAME", "auto"),
            provider="openai",
            api_base=os.environ["AGL_OPENAI_BASE_URL"],
            api_key=agl_key or "agent-lightning",
            temperature=1.0,
            thinking=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            max_retries=12,
            request_timeout_seconds=300,
            sdk_max_retries=0,
            max_concurrent_per_key=1,
            global_max_concurrency=1,
        )
        perception_enabled = _env_flag("TRAINING_PERCEPTION_ENABLED")
        output = root / "output"
        workspace = root / "workspace"
        staging = root / "staging"
        common = {
            "output_dir": output,
            "workspace_dir": workspace,
            "staging_dir": staging,
            "workflow": "react",
            "llm": llm,
            "perception_enabled": perception_enabled,
            "perception_llm": llm if perception_enabled else None,
            "max_steps": max_steps,
            "protocol_mode": "strict_retry",
            **context_settings,
            "concurrency": 1,
            "timeout_seconds": timeout,
            "local_isolation": "bubblewrap",
        }
        if source == "dare_bench":
            from experiments.dare_bench.runner import (
                DareDataStore,
                RunSettings,
                load_tasks,
                run_tasks,
                select_tasks,
            )

            tasks = select_tasks(load_tasks(Path(os.environ["DARE_MANIFEST"])), task_ids=[task_id])
            return await run_tasks(
                tasks,
                RunSettings(
                    data_store=DareDataStore(databases_dir=Path(os.environ["DARE_DATABASES_DIR"])),
                    memory_mb=8192,
                    cpu_cores=float(threads),
                    sandbox_python=Path(os.environ["DARE_PYTHON"]),
                    **common,
                ),
            )
        if source == "data_agent_rl":
            from experiments.data_agent_rl.runner import (
                BucketDataStore,
                RunSettings,
                load_task,
                run_tasks,
            )

            tasks = [load_task(Path(os.environ["DATA_AGENT_ROOT"]), task_id)]
            return await run_tasks(
                tasks,
                RunSettings(
                    data_store=BucketDataStore(
                        cache_root=Path(os.environ["DATA_AGENT_CACHE"]), offline=True
                    ),
                    download_concurrency=1,
                    sandbox_python=Path(os.environ["GENERAL_PYTHON"]),
                    compute_threads=threads,
                    **common,
                ),
            )
        if source == "datamind_sql":
            from experiments.datamind_sql.runner import (
                DataMindSQLDataStore,
                RunSettings,
                load_tasks,
                run_tasks,
                select_tasks,
            )

            tasks = select_tasks(
                load_tasks(Path(os.environ["TRAINING_MIX_MANIFEST"])), task_ids=[task_id]
            )
            return await run_tasks(
                tasks,
                RunSettings(
                    data_store=DataMindSQLDataStore(Path(os.environ["DATAMIND_ROOT"])),
                    sandbox_python=Path(os.environ["GENERAL_PYTHON"]),
                    compute_threads=threads,
                    **common,
                ),
            )
        raise ValueError(f"Unsupported training source: {source}")

    @staticmethod
    async def _post_reward(
        event_url: str,
        agl_key: str,
        source: str,
        task_id: str,
        reward: float,
        *,
        scorer: str = "programmatic",
        details: dict[str, Any] | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {agl_key}"}
        event = {
            "event_type": "reward",
            "data": {
                "value": reward,
                "source": source,
                "task_id": task_id,
                "scorer": scorer,
                **(details or {}),
            },
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            for attempt in range(1, 4):
                try:
                    response = await client.post(event_url, json=event, headers=headers)
                    response.raise_for_status()
                    return
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(attempt)


__all__ = ["TrainingMixAgent", "_env_flag", "reward_from_result"]
