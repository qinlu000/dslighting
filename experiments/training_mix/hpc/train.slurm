#!/usr/bin/env bash
#SBATCH --job-name=dslight-grpo
#SBATCH --partition=acd_ue
#SBATCH --nodelist=ACD1-[8,13,25,30]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/train-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
module load singularity/4.4.1

export PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-/data/user/qinlucao/datasets/training_mix_2000_bundle_v1}"
export CONTROLLER_ENV="${CONTROLLER_ENV:-/data/user/qinlucao/envs/dslighting-agl-controller-tmix-v1}"
export TRAINER_ENV="${TRAINER_ENV:-/data/user/qinlucao/envs/agent-lightning-trainer-cu130-tmix-v1}"
export MIN_NVIDIA_DRIVER_MAJOR="${MIN_NVIDIA_DRIVER_MAJOR:-580}"
export TRAINER_IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
export DARE_PYTHON="${DARE_PYTHON:-/data/user/qinlucao/envs/darebench310-tmix-v1/bin/python}"
export GENERAL_PYTHON="${GENERAL_PYTHON:-/data/user/qinlucao/envs/data-agent312-tmix-v1/bin/python}"
: "${MODEL_PATH:?Pass MODEL_PATH with sbatch --export=ALL,MODEL_PATH=/absolute/model/path}"
export MODEL_PATH
export RUN_ROOT="${RUN_ROOT:-/data/user/qinlucao/runs/training_mix/${SLURM_JOB_ID}}"
export TRAINING_GPUS="${TRAINING_GPUS:-8}"
export TRAINING_MAX_STEPS="${TRAINING_MAX_STEPS:-10}"
export TRAINING_COMPUTE_THREADS="${TRAINING_COMPUTE_THREADS:-2}"
export TRAINING_ROLLOUT_TIMEOUT_SECONDS="${TRAINING_ROLLOUT_TIMEOUT_SECONDS:-1800}"
export TRAINING_AGENT_DEADLINE_SECONDS="${TRAINING_AGENT_DEADLINE_SECONDS:-1650}"
export ROLLOUT_CONCURRENCY="${ROLLOUT_CONCURRENCY:-32}"

echo "job=$SLURM_JOB_ID node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} run=$RUN_ROOT"
if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader; then
  echo "warning: nvidia-smi inventory query returned non-zero; CUDA preflight will decide readiness" >&2
fi

exec bash "$PROJECT_ROOT/experiments/training_mix/hpc/run_agent_lightning.sh"
