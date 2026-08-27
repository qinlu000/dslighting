#!/usr/bin/env bash
#SBATCH --job-name=dslight-ood-preflight
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/ood-preflight-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/data/user/qinlucao/envs/dslighting-agl-controller-tmix-v1}"
GENERAL_PYTHON="${GENERAL_PYTHON:-/data/user/qinlucao/envs/data-agent312-tmix-v1/bin/python}"
AGENTIC_PYTHON="${AGENTIC_PYTHON:-/data/user/qinlucao/envs/agenticdatabench310-tmix-v1/bin/python}"
DABENCH_ROOT="${DABENCH_ROOT:-/data/user/qinlucao/datasets/DABench/data}"
AGENTIC_ROOT="${AGENTIC_ROOT:-/data/user/qinlucao/datasets/AgenticDataBench}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
"$CONTROLLER_ENV/bin/python" -m experiments.training_mix.hpc.preflight_ood \
  --controller-python "$CONTROLLER_ENV/bin/python" \
  --dabench-python "$GENERAL_PYTHON" \
  --agentic-python "$AGENTIC_PYTHON" \
  --dabench-root "$DABENCH_ROOT" \
  --agentic-root "$AGENTIC_ROOT"
