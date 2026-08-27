from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training_mix.hpc.preflight_compute import _model
from experiments.training_mix.preflight import load_manifest


def test_load_manifest_requires_hidden_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "source": "datamind_sql",
                "task_id": "task_1",
                "evaluator": {"kind": "csv_table_match", "agent_visible": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_manifest(path)[0]["task_id"] == "task_1"

    path.write_text(
        json.dumps(
            {
                "source": "datamind_sql",
                "task_id": "task_1",
                "evaluator": {"kind": "csv_table_match", "agent_visible": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="visibility"):
        load_manifest(path)


def test_compute_preflight_requires_complete_model_shards(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}})
    )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")

    assert _model(tmp_path)["passed"] is True

    shard.unlink()
    report = _model(tmp_path)
    assert report["passed"] is False
    assert "model-00001-of-00001.safetensors" in report["failures"][0]
