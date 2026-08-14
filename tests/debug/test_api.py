from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_debug_probe(profile: str, *, set_raw_before_init: bool = False) -> dict:
    code = f"""
        import asyncio
        import json
        import logging
        import os

        import litellm

        from dslighting.debug.api import get_debug_session, init_debug
        from dslighting.debug.litellm_bridge import DSLightingLiteLLMLogger

        if {set_raw_before_init!r}:
            litellm.log_raw_request_response = True

        init_debug(enabled=True, profile={profile!r}, console_output=False)
        print(json.dumps({{
            "litellm_log": os.environ["LITELLM_LOG"],
            "httpx_level": logging.getLogger("httpx").level,
            "httpcore_level": logging.getLogger("httpcore").level,
            "litellm_level": logging.getLogger("LiteLLM").level,
            "raw": litellm.log_raw_request_response,
            "has_bridge": any(
                isinstance(callback, DSLightingLiteLLMLogger)
                and callback.provider_raw == ({profile!r} == "provider_raw")
                for callback in litellm.callbacks
            ),
        }}))

        session = get_debug_session()
        if session is not None:
            asyncio.run(session.close())
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_init_debug_suppresses_noisy_loggers_and_registers_bridge() -> None:
    result = _run_debug_probe("full", set_raw_before_init=True)

    assert result == {
        "litellm_log": "WARNING",
        "httpx_level": 30,
        "httpcore_level": 30,
        "litellm_level": 30,
        "raw": False,
        "has_bridge": True,
    }


def test_provider_raw_keeps_litellm_debug_but_suppresses_http_stack() -> None:
    result = _run_debug_probe("provider_raw")

    assert result == {
        "litellm_log": "DEBUG",
        "httpx_level": 30,
        "httpcore_level": 30,
        "litellm_level": 10,
        "raw": True,
        "has_bridge": True,
    }
