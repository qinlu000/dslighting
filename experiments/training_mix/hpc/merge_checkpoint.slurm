#!/usr/bin/env bash
#SBATCH --job-name=dslight-merge
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/merge-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
module load singularity/4.4.1

TRAINER_ENV="${TRAINER_ENV:-/data/user/qinlucao/envs/agent-lightning-trainer-cu130-tmix-v1}"
TRAINER_IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
: "${CHECKPOINT_ROOT:?Pass the run checkpoint directory with CHECKPOINT_ROOT=/absolute/path/checkpoints}"

marker="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
test -f "$marker"
step="$(tr -d '[:space:]' < "$marker")"
if [[ ! "$step" =~ ^[0-9]+$ ]]; then
  echo "Invalid checkpoint marker: $marker" >&2
  exit 2
fi

actor_dir="$CHECKPOINT_ROOT/global_step_${step}/actor"
target_dir="${TARGET_DIR:-$CHECKPOINT_ROOT/hf_global_step_${step}}"
test -f "$actor_dir/fsdp_config.json"
if [[ -e "$target_dir" ]]; then
  echo "Refusing to overwrite existing target: $target_dir" >&2
  exit 2
fi

export SINGULARITYENV_DSLIGHTING_TRAINER_ENV="$TRAINER_ENV"
export SINGULARITYENV_DSLIGHTING_ACTOR_DIR="$actor_dir"
export SINGULARITYENV_DSLIGHTING_TARGET_DIR="$target_dir"
singularity exec --nv --bind /data/user/qinlucao:/data/user/qinlucao \
  "$TRAINER_IMAGE" /bin/bash -c \
  'exec "$DSLIGHTING_TRAINER_ENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir "$DSLIGHTING_ACTOR_DIR" --target_dir "$DSLIGHTING_TARGET_DIR" --use_cpu_initialization'

test -f "$target_dir/config.json"
find "$target_dir" -maxdepth 1 -type f ! -name MODEL_SHA256SUMS -printf '%f\0' \
  | sort -z | (cd "$target_dir" && xargs -0 sha256sum) > "$target_dir/MODEL_SHA256SUMS"
echo "merged_checkpoint=$target_dir step=$step"
