"""Launch GRPO for the frozen training mixture through Agent Lightning v1."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

AGENT_LIGHTNING_COMMIT = "8435586d147b4cf7bff33e687d7317149e79cbb8"


def load_rollout_rows(path: Path) -> list[dict[str, str]]:
    """Strip evaluator fields before records enter the rollout API."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in path.expanduser().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source, task_id = str(record["source"]), str(record["task_id"])
        key = (source, task_id)
        if key in seen:
            raise ValueError(f"Duplicate rollout task: {key}")
        seen.add(key)
        rows.append(
            {
                "source": source,
                "task_id": task_id,
                "data_source": "dslighting_training_mix",
            }
        )
    if not rows:
        raise ValueError("Training manifest is empty")
    return rows


def balanced_prefix(
    rows: Sequence[dict[str, str]], limit: int | None
) -> list[dict[str, str]]:
    """Select a deterministic source-balanced prefix for smoke and pilot runs."""

    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("limit_tasks must be positive")
    if limit >= len(rows):
        return list(rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    selected: list[dict[str, str]] = []
    source_names = sorted(grouped)
    for index in range(max(len(values) for values in grouped.values())):
        for source in source_names:
            if index < len(grouped[source]):
                selected.append(grouped[source][index])
                if len(selected) == limit:
                    return selected
    return selected


def verl_overrides(
    *,
    model: str,
    output_dir: str,
    gpus: int,
    train_batch_size: int,
    group_size: int,
    tensor_parallel_size: int,
    ray_num_cpus: int,
    total_epochs: int,
    save_freq: int,
    agl_base_url: str,
    agl_key: str,
    run_name: str,
    rollout_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    if min(
        gpus,
        train_batch_size,
        group_size,
        tensor_parallel_size,
        ray_num_cpus,
        total_epochs,
        save_freq,
        rollout_timeout_seconds,
    ) <= 0:
        raise ValueError(
            "GPU, batch, group, tensor-parallel, epoch, and save-frequency values "
            "must be positive"
        )
    if gpus % tensor_parallel_size:
        raise ValueError("tensor_parallel_size must divide gpus")
    return {
        "ray_kwargs": {"ray_init": {"num_cpus": ray_num_cpus}},
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "enable_rollout_level_advantage": True,
            "rollout_correction": {
                "rollout_is": "token",
                "rollout_is_threshold": 2,
            },
        },
        "data": {
            "train_batch_size": train_batch_size,
            "dataloader_num_workers": 0,
            "max_prompt_length": 8192,
            "max_response_length": 24576,
            "filter_overlong_prompts": False,
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "n": group_size,
                "tensor_model_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 8192 + 24576,
                "log_prob_micro_batch_size_per_gpu": 1,
            },
            "actor": {
                "ppo_mini_batch_size": train_batch_size,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "entropy_coeff": 0,
                "policy_loss": {"loss_mode": "per_rollout_mean"},
            },
            "ref": {"log_prob_micro_batch_size_per_gpu": 1},
            "model": {
                "path": model,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": gpus,
            "nnodes": 1,
            "val_before_train": False,
            "test_freq": -1,
            "save_freq": save_freq,
            "default_local_dir": output_dir,
            "total_epochs": total_epochs,
            "logger": ["console"],
            "project_name": "dslighting",
            "experiment_name": run_name,
        },
        "agentlightning": {
            "agl_base_url": agl_base_url,
            "agl_key": agl_key,
            "hooks": str(Path(__file__).resolve().parent / "hpc" / "rollout_audit_hook.py"),
            "rollout_timeout_seconds": rollout_timeout_seconds,
            "max_ppo_update_times": 2,
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 8192,
                "trajectory_max_response_length": 24576,
            },
            "async_rollout": {
                # Train on every submitted group. Agent Lightning's buffered
                # mode takes the first groups to finish, which biases this
                # heterogeneous mixture toward faster task families.
                "enabled": False,
            },
            "local": {
                "agent_class": "experiments.training_mix.agent:TrainingMixAgent",
                "env_map": {
                    "TRAINING_SOURCE": "input.source",
                    "TRAINING_TASK_ID": "input.task_id",
                },
            },
        },
    }


def build_config(overrides: dict[str, Any]) -> Any:
    import importlib.resources

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    package = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(package), version_base=None):
        base = compose(config_name="config")
    OmegaConf.set_struct(base, False)
    return OmegaConf.merge(base, OmegaConf.create(overrides))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    # Forty divides the 2,000-task full mixture, so VERL's drop_last=True does
    # not silently omit tasks from a one-epoch run.
    parser.add_argument("--train-batch-size", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--ray-num-cpus", type=int, default=32)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--agl-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--agl-key", default=os.getenv("AGL_KEY", ""))
    parser.add_argument("--run-name", default="training_mix_grpo")
    parser.add_argument("--rollout-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--limit-tasks",
        type=int,
        help="Use a deterministic source-balanced subset for smoke or pilot runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = balanced_prefix(load_rollout_rows(args.manifest), args.limit_tasks)
    overrides = verl_overrides(
        model=args.model,
        output_dir=str(args.output_dir.expanduser().resolve()),
        gpus=args.gpus,
        train_batch_size=args.train_batch_size,
        group_size=args.group_size,
        tensor_parallel_size=args.tensor_parallel_size,
        ray_num_cpus=args.ray_num_cpus,
        total_epochs=args.total_epochs,
        save_freq=args.save_freq,
        agl_base_url=args.agl_base_url,
        agl_key=args.agl_key,
        run_name=args.run_name,
        rollout_timeout_seconds=args.rollout_timeout_seconds,
    )
    if args.dry_run:
        printable = json.loads(json.dumps(overrides))
        printable["agentlightning"]["agl_key"] = "<set>" if args.agl_key else ""
        print(
            json.dumps(
                {
                    "agent_lightning_commit": AGENT_LIGHTNING_COMMIT,
                    "tasks": len(rows),
                    "source_counts": {
                        source: sum(row["source"] == source for row in rows)
                        for source in sorted({row["source"] for row in rows})
                    },
                    "ground_truth_in_rollout_input": False,
                    "config_overrides": printable,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from agentlightning.verl.entrypoint import run_ppo

    config = build_config(overrides)
    # Agent Lightning requires a non-empty validation sequence. Evaluation is
    # disabled above; this placeholder is never used for checkpoint selection.
    run_ppo(config, train_dataset=rows, val_dataset=rows[:1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
