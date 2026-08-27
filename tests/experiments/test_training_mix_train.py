from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training_mix.train import balanced_prefix, load_rollout_rows, verl_overrides


def test_rollout_rows_strip_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "source": "data_agent_rl",
                "task_id": "task_1",
                "question": "secret question",
                "evaluator": {"ground_truth": "SECRET"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_rollout_rows(path)

    assert rows == [
        {
            "source": "data_agent_rl",
            "task_id": "task_1",
            "data_source": "dslighting_training_mix",
        }
    ]
    assert "SECRET" not in json.dumps(rows)


def test_verl_config_uses_rollout_level_grpo() -> None:
    config = verl_overrides(
        model="model",
        output_dir="/tmp/checkpoints",
        gpus=8,
        train_batch_size=32,
        group_size=4,
        tensor_parallel_size=1,
        ray_num_cpus=32,
        total_epochs=1,
        save_freq=1,
        agl_base_url="http://127.0.0.1:8080",
        agl_key="",
        run_name="test",
    )

    assert config["algorithm"]["adv_estimator"] == "grpo"
    assert config["ray_kwargs"]["ray_init"]["num_cpus"] == 32
    assert config["data"]["dataloader_num_workers"] == 0
    assert config["algorithm"]["enable_rollout_level_advantage"] is True
    assert config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "per_rollout_mean"
    assert config["agentlightning"]["async_rollout"] == {"enabled": False}
    assert config["actor_rollout_ref"]["rollout"]["max_model_len"] == 32768
    assert config["trainer"]["default_local_dir"] == "/tmp/checkpoints"
    assert config["agentlightning"]["hooks"].endswith("/hpc/rollout_audit_hook.py")
    assert config["agentlightning"]["rollout_timeout_seconds"] == 1800
    assert config["trainer"]["save_freq"] == 1


def test_rollout_timeout_is_configurable() -> None:
    config = verl_overrides(
        model="model",
        output_dir="/tmp/checkpoints",
        gpus=4,
        train_batch_size=40,
        group_size=4,
        tensor_parallel_size=1,
        ray_num_cpus=16,
        total_epochs=1,
        save_freq=10,
        agl_base_url="http://127.0.0.1:8080",
        agl_key="",
        run_name="test",
        rollout_timeout_seconds=900,
    )

    assert config["agentlightning"]["rollout_timeout_seconds"] == 900


def test_balanced_prefix_is_deterministic_across_sources() -> None:
    rows = [
        {"source": source, "task_id": f"{source}_{index}"}
        for source in ("dare_bench", "data_agent_rl", "datamind_sql")
        for index in range(3)
    ]

    selected = balanced_prefix(rows, 5)

    assert [(row["source"], row["task_id"]) for row in selected] == [
        ("dare_bench", "dare_bench_0"),
        ("data_agent_rl", "data_agent_rl_0"),
        ("datamind_sql", "datamind_sql_0"),
        ("dare_bench", "dare_bench_1"),
        ("data_agent_rl", "data_agent_rl_1"),
    ]

    with pytest.raises(ValueError, match="positive"):
        balanced_prefix(rows, 0)
