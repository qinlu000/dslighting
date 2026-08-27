"""Persist compact rollout requests, responses, and rewards before AGL deletes them."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agentlightning.hooks import RolloutHooks

from experiments.training_mix.protocol_reward import infer_agent_role_from_request


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _compact_event(event: Any) -> dict[str, Any]:
    payload = _json(event)
    data = dict(payload.get("data") or {})
    if payload.get("event_type") == "model_request":
        request = dict(data.get("request") or {})
        response = dict(data.get("response") or {})
        data["request"] = {
            key: request[key]
            for key in (
                "model",
                "messages",
                "temperature",
                "thinking",
                "chat_template_kwargs",
            )
            if key in request
        }
        data["agent_role"] = infer_agent_role_from_request(request)
        data["response"] = {
            "choices": [
                {key: choice[key] for key in ("index", "message", "finish_reason") if key in choice}
                for choice in response.get("choices") or []
            ],
            "usage": response.get("usage"),
        }
        data = {
            key: data[key]
            for key in (
                "status",
                "http_status",
                "latency_ms",
                "finish_reason",
                "request",
                "response",
                "agent_role",
            )
            if key in data
        }
    payload["data"] = data
    return payload


class TrainingMixAuditHooks(RolloutHooks):
    def on_startup(self, store: Any | None = None) -> None:
        value = os.environ.get("TRAINING_ROLLOUT_AUDIT_PATH", "").strip()
        if not value:
            raise RuntimeError("TRAINING_ROLLOUT_AUDIT_PATH is required")
        self.path = Path(value).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _write(self, state: str, rollout: Any, events: dict[str, list[Any]]) -> None:
        record = {
            "state": state,
            "rollout": _json(rollout),
            "events_by_attempt": {
                attempt: [_compact_event(event) for event in values]
                for attempt, values in events.items()
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def on_succeeded(self, rollout: Any, events: dict[str, list[Any]], store: Any) -> None:
        self._write("succeeded", rollout, events)

    def on_failed(self, rollout: Any, store: Any) -> None:
        self._write("failed", rollout, {})
