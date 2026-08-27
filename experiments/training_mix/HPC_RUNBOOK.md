# HPC GRPO runbook

The local checkout is the source of truth. The HPC run uses three immutable
inputs:

- project snapshot: `/data/user/qinlucao/research/dslighting-training-mix-20260826-final`;
- task bundle: `/data/user/qinlucao/datasets/training_mix_2000_bundle_v1`;
- offline wheelhouse: `/data/user/qinlucao/datasets/training_mix_hpc_wheelhouse_v1`.

The wheelhouse contains CPython 3.10.12 and 3.12.13, exact dependency locks,
the pinned Agent Lightning 1.0.1 wheel, and SHA-256 checksums. Environment setup
does not access GitHub, PyPI, or Hugging Face.

Keep the runtime roles separate:

- host Python 3.12 controller: Agent Lightning server/controller and DSLighting;
- Singularity Python 3.12 trainer: Torch 2.11/cu130, vLLM 0.20.2, VERL 0.8.0,
  Agent Lightning 1.0.1, and FlashAttention 2.8.3;
- host task interpreters: DARE Python 3.10 and general data-task Python 3.12.

The trainer must run in
`/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif`. Its Ubuntu
22.04/glibc 2.35 runtime matches the official vLLM wheel. The HPC host has
glibc 2.28 and cannot load that wheel directly. Bubblewrap remains on the host
and uses the two task interpreters; it is not nested inside Singularity.
Every GPU Slurm entrypoint explicitly loads `singularity/4.4.1`; compute nodes
do not expose a container runtime in `PATH` until its module is loaded.
HPC3's DPC wrapper also mistakes existing directory arguments after the image
for sandbox images. Model, checkpoint, and output paths are therefore expanded
from environment variables only after the container starts.

Build only the host controller and task-sandbox environments in a CPU-only
compute allocation:

```bash
sbatch experiments/training_mix/hpc/setup_envs.slurm
```

Build and validate the current CUDA 13, Torch 2.11, VERL 0.8.0, and vLLM 0.20.2
trainer separately:

```bash
sbatch experiments/training_mix/hpc/setup_trainer_vllm0202.slurm
```

The scripts write environment fingerprints under
`/data/user/qinlucao/runs/training_mix/environment`. SFT uses its independent
LLaMA-Factory environment, created only when SFT must be reproduced:

```bash
sbatch experiments/training_mix/hpc/setup_sft_env.slurm
```

Then run four gates:

1. Compute-node preflight:

   ```bash
   sbatch --export=ALL,MODEL_PATH=/absolute/model/path \
     experiments/training_mix/hpc/preflight.slurm
   ```

   This verifies every bundle hash, all 2,000 deterministic evaluator inputs,
   actual trainer imports/CUDA visibility inside Singularity, and real
   Bubblewrap execution with both task interpreters.

2. One-update balanced smoke:

   ```bash
   sbatch --gres=gpu:2 --cpus-per-task=24 --mem=256G --time=02:00:00 \
     --export=ALL,MODEL_PATH=/absolute/model/path,TRAINING_GPUS=2,LIMIT_TASKS=32,TRAIN_BATCH_SIZE=32,RUN_NAME=smoke32,SAVE_FREQ=1,ROLLOUT_CONCURRENCY=16 \
     experiments/training_mix/hpc/train.slurm
   ```

3. Balanced pilot:

   ```bash
   sbatch --gres=gpu:4 --cpus-per-task=48 --mem=384G --time=04:00:00 \
     --export=ALL,MODEL_PATH=/absolute/model/path,TRAINING_GPUS=4,LIMIT_TASKS=96,TRAIN_BATCH_SIZE=32,RUN_NAME=pilot96,SAVE_FREQ=3,ROLLOUT_CONCURRENCY=24 \
     experiments/training_mix/hpc/train.slurm
   ```

4. Full 2,000-task run after checking smoke/pilot rollout failures, GPU/CPU
   utilization, reward distributions, checkpoint creation, and wall time:

   ```bash
   sbatch --export=ALL,MODEL_PATH=/absolute/model/path,RUN_NAME=training_mix_grpo_full2000,RUN_ROOT=/data/user/qinlucao/runs/training_mix/grpo-full2000-v1 \
     experiments/training_mix/hpc/train.slurm
   ```

   `RUN_ROOT` is a stable experiment directory, not a Slurm job directory.
   Keep it unchanged when resubmitting an interrupted run. VERL's
   `resume_mode=auto` reads `latest_checkpointed_iteration.txt`, restores the
   model, optimizer, and dataloader state, and the rollout audit appends to the
   existing JSONL. Slurm job IDs remain in `train-%j.out` logs.

Every successful run writes a final VERL FSDP checkpoint under its `RUN_ROOT`.
Before Agent Lightning removes completed server records, a compact JSONL audit
at `RUN_ROOT/rollout_audit.jsonl` preserves task IDs, model messages, thinking
flags, timings, task rewards, strict ReAct protocol rewards, and combined
rewards without token-level log-prob payloads. The default combined reward is
`0.9 * task_reward + 0.1 * protocol_reward`; set
`TRAINING_FORMAT_REWARD_WEIGHT` to override the protocol weight.
Convert the selected checkpoint to a normal Hugging Face model before OOD
evaluation:

```bash
sbatch --export=ALL,CHECKPOINT_ROOT=/absolute/run/checkpoints \
  experiments/training_mix/hpc/merge_checkpoint.slurm
```

The merge refuses to overwrite an existing model and writes
`MODEL_SHA256SUMS` beside the Hugging Face shards.

Validate model serving plus one real task from each OOD benchmark before a
full evaluation:

```bash
sbatch --export=ALL,MODEL_DIR=/absolute/merged/hf/model \
  experiments/training_mix/hpc/ood_smoke.slurm
```

This runs DABench's deterministic DSLighting grader and AgenticDataBench's
official evaluator. Both model calls and training rollouts explicitly disable
Qwen chat-template thinking while retaining the visible ReAct `<Think>` turns.
The smoke is an infrastructure gate, so the official evaluator still runs and
records a zero if the model fails to produce its requested artifact; model
failure is not misreported as an evaluator crash.

Run the complete 503-task public OOD evaluation with the same entrypoint after
the smoke passes:

```bash
sbatch --cpus-per-task=12 --mem=128G --time=24:00:00 \
  --export=ALL,MODEL_DIR=/absolute/merged/hf/model,OOD_FULL=1,OOD_CONCURRENCY=4 \
  experiments/training_mix/hpc/ood_smoke.slurm
```

Full mode evaluates all 257 DABench tasks and all 246 public
AgenticDataBench tasks. It writes `ood_full_summary.json`, the DABench result
CSV/metadata, the DSLighting AgenticDataBench run summary, and the unmodified
official AgenticDataBench score JSON. A model that produces a wrong or missing
artifact receives its programmatic zero; missing result rows or evaluator
artifacts fail the job as an infrastructure error.
The one-GPU partition contract permits at most 12 CPUs per GPU. Concurrency 4
therefore leaves eight compute threads for four two-thread Bubblewrap sandboxes
and the remaining CPUs for the controller and vLLM server.

The default full-run batch is 40, which divides all 2,000 tasks exactly. The
8-GPU job requests the cluster maximum of 96 CPUs and executes 32 rollouts
concurrently. Every Bubblewrap sandbox is limited to two compute threads and
ten minutes per code execution. Training
waits for every submitted rollout
group instead of selecting the fastest completions, so heterogeneous task
runtimes cannot change the source distribution. Transport, API, and sandbox
failures are raised as failed rollouts rather than silently converted to reward
zero. Rollout requests disable provider-internal thinking; DSLighting's visible
ReAct `<Think>` turns are unchanged.

Ray is explicitly limited to four CPUs per training GPU (32 CPUs for the
default 8-GPU run). This avoids Ray detecting all CPUs on a shared Slurm node
and leaves 64 CPUs for the 32 two-thread local rollout sandboxes.
Torch, Triton, CuPy, vLLM, Hugging Face, and XDG caches are explicitly rooted
in a job-specific node-local `/tmp` directory. Generated compiler files do not
cross the DPC shared filesystem, and the container never attempts to write
into the read-only Singularity module installation. The launcher removes only
its job-specific cache directory on exit.

Serve the merged Hugging Face directory through an OpenAI-compatible vLLM
endpoint, then evaluate it without changing the training manifest. DABench and
AgenticDataBench are OOD-only benchmarks and use their own official
deterministic graders through the DSLighting adapters.
