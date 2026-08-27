from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.training_mix.rollout import _llm, _python_path


def test_python_path_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    interpreter = tmp_path / "environment" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)

    parsed = _python_path(str(interpreter))

    assert parsed == interpreter.absolute()
    assert parsed != interpreter.resolve()


def test_remote_teacher_disables_thinking_without_vllm_only_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret")

    config = _llm(
        SimpleNamespace(
            model="openai/DeepSeek-V4-Flash",
            api_base="https://gateway.example/v1",
            api_key=None,
            temperature=0.0,
            concurrency=8,
        )
    )

    assert config.thinking is False
    assert config.extra_body == {}
    assert config.api_key == "secret"
