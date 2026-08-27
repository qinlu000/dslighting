# Research training pipeline

The canonical task input is
`paper/training_mix/artifacts/manifests/training_mix_2000.jsonl` (2,000 tasks). The
runtime has three deliberately small layers:

1. source adapters in `experiments/dare_bench`, `experiments/data_agent_rl`,
   and `experiments/datamind_sql` stage inputs and compute deterministic reward;
2. `TrainingMixAgent` runs exactly one DSLighting ReAct rollout in Bubblewrap;
3. Agent Lightning v1 captures the OpenAI-compatible model calls and hands
   trajectories/reward to VERL GRPO.

The code is intentionally flat:

- `experiments/common.py` contains only result I/O, run locking, bounded
  concurrency, CPU thread variables, and DSLighting runner construction.
- each source `runner.py` keeps its own task schema, staging, prompt, and
  deterministic grader;
- `experiments/training_mix/rollout.py` is the local teacher entry point;
- `experiments/training_mix/agent.py` is the one-task Agent Lightning bridge.

There is no benchmark registry, adapter inheritance tree, or dynamic dispatch.

The rollout reward is deterministic and uses no LLM judge:

```text
reward = 0.9 * task_reward + 0.1 * protocol_reward
protocol_reward = 0.8 * strict_valid_turn_rate + 0.2 * valid_terminal_answer
```

Protocol reward is computed from the raw model responses recorded by Agent
Lightning, before DSLighting performs any conservative response repair. A
strictly valid turn contains one non-empty `<Think>` block followed by exactly
one `<Action>` or `<Answer>` block, with no surrounding text. Actions contain
one non-empty fenced Python block; answers are non-empty plain text. Override
the default protocol weight with `TRAINING_FORMAT_REWARD_WEIGHT` if needed.

Set `TRAINING_PERCEPTION_ENABLED=1` for joint Solver/Perception RL. Solver and
Perception calls use the same policy but independent conversation contexts.
The role is inferred from each captured system prompt: Solver turns accept
`<Action>`, `<Explore>`, or `<Answer>`, while Perception turns accept `<Action>`
or `<Report>`. Both roles receive the same deterministic episode reward, and
the rollout audit stores `agent_role` on every model request. The default is
`0`, preserving the Solver-only baseline.

Provider-internal thinking is disabled on rollout requests. The ReAct
workflow's visible `<Think>` protocol remains part of the trained trajectory.

Ground truth is stripped from Agent Lightning rollout inputs and is never
mounted into the code sandbox. Successful rollout scratch directories are
temporary. DABench and AgenticDataBench remain external OOD evaluations.

DataMind contributes only the curated SQL adapter in `experiments/datamind_sql`.
The earlier standalone DataMind Perception SFT/LLM-judge experiment is not part
of this pipeline.

The current role-aware SFT input is the frozen 511-row dataset at
`paper/training_mix/artifacts/cold_start/perception_sft_final_medium_t09_qwen3_24k`.
It contains 276 Solver and 235 Perception trajectories and is shared by the
Qwen3-4B and Qwen3-8B role-aware checkpoints. The older 207-row Solver-only
dataset remains solely as the cold-start baseline for the ongoing 4B GRPO run.

Run the local data/environment gate before any rollout:

```bash
.venv/bin/python -m experiments.training_mix \
  --manifest paper/training_mix/artifacts/manifests/training_mix_2000.jsonl \
  --dare-manifest /data/caoqinlu/datasets/dare-bench/derived/fixed_gt_train/train_fixed_gt.jsonl \
  --dare-databases-dir /data/caoqinlu/datasets/dare-bench/huggingface/train/databases \
  --data-agent-root /data/caoqinlu/datasets/data_agent_rl_environment_train \
  --data-agent-cache /data/caoqinlu/datasets/data_agent_rl_environment_train_bucket_cache \
  --datamind-root /data/caoqinlu/datasets/DataMind-Data \
  --dare-python /data/caoqinlu/datasets/dare-bench/repo/.venv/bin/python \
  --general-python /data/caoqinlu/datasets/data_agent_rl_environment_train/.venv/bin/python
```

The trainer is pinned to Agent Lightning commit
`8435586d147b4cf7bff33e687d7317149e79cbb8` (v1.0.1). Its official setup
script installs the matching VERL/vLLM/CUDA stack; do not reuse or mutate the
DSLighting controller environment. A local static check does not require the
GPU stack:

```bash
.venv/bin/python -m experiments.training_mix.train \
  --manifest paper/training_mix/artifacts/manifests/training_mix_2000.jsonl \
  --model /path/to/model \
  --output-dir /path/to/run/checkpoints \
  --dry-run
```

For HPC transfer, materialize the selected files rather than copying all three
source datasets. The bundle uses hard links on the local filesystem and stores
per-file SHA-256 values for transfer verification:

```bash
.venv/bin/python -m experiments.training_mix.materialize_bundle \
  --manifest paper/training_mix/artifacts/manifests/training_mix_2000.jsonl \
  --dare-manifest paper/training_mix/artifacts/manifests/dare_bench_600.jsonl \
  --dare-databases-dir /data/caoqinlu/datasets/dare-bench/huggingface/train/databases \
  --data-agent-root /data/caoqinlu/datasets/data_agent_rl_environment_train \
  --data-agent-cache /data/caoqinlu/datasets/data_agent_rl_environment_train_bucket_cache \
  --datamind-root /data/caoqinlu/datasets/DataMind-Data \
  --output /data/caoqinlu/datasets/training_mix_2000_bundle_v1
```
