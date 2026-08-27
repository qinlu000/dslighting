#!/usr/bin/env bash
#SBATCH --job-name=dslight-grpo-preflight
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/preflight-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
module load singularity/4.4.1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/data/user/qinlucao/datasets/training_mix_2000_bundle_v1}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/data/user/qinlucao/envs/dslighting-agl-controller-tmix-v1}"
TRAINER_ENV="${TRAINER_ENV:-/data/user/qinlucao/envs/agent-lightning-trainer-cu130-tmix-v1}"
TRAINER_IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
DARE_PYTHON="${DARE_PYTHON:-/data/user/qinlucao/envs/darebench310-tmix-v1/bin/python}"
GENERAL_PYTHON="${GENERAL_PYTHON:-/data/user/qinlucao/envs/data-agent312-tmix-v1/bin/python}"
: "${MODEL_PATH:?Pass MODEL_PATH with sbatch --export=ALL,MODEL_PATH=/absolute/model/path}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "job=$SLURM_JOB_ID node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

"$CONTROLLER_ENV/bin/python" -m experiments.training_mix.hpc.preflight_compute \
  --bundle-root "$BUNDLE_ROOT" \
  --controller-python "$CONTROLLER_ENV/bin/python" \
  --trainer-python "$TRAINER_ENV/bin/python" \
  --trainer-image "$TRAINER_IMAGE" \
  --container-bind /data/user/qinlucao \
  --dare-python "$DARE_PYTHON" \
  --general-python "$GENERAL_PYTHON" \
  --model "$MODEL_PATH" \
  --verify-bundle-hashes

"$CONTROLLER_ENV/bin/python" -m experiments.training_mix \
  --manifest "$BUNDLE_ROOT/manifests/train_full.jsonl" \
  --dare-manifest "$BUNDLE_ROOT/dare/tasks.jsonl" \
  --dare-databases-dir "$BUNDLE_ROOT/dare/databases" \
  --data-agent-root "$BUNDLE_ROOT/data_agent/dataset" \
  --data-agent-cache "$BUNDLE_ROOT/data_agent/cache" \
  --datamind-root "$BUNDLE_ROOT/datamind" \
  --dare-python "$DARE_PYTHON" \
  --general-python "$GENERAL_PYTHON"

preflight_argv_json="$("$CONTROLLER_ENV/bin/python" -c \
  'import json, sys; print(json.dumps(sys.argv[1:]))' \
  --manifest "$BUNDLE_ROOT/manifests/train_full.jsonl" \
  --model "$MODEL_PATH" \
  --output-dir "/data/user/qinlucao/runs/training_mix/preflight-${SLURM_JOB_ID}/checkpoints" \
  --gpus 8 \
  --limit-tasks 32 \
  --dry-run)"
SINGULARITYENV_PYTHONPATH="$PYTHONPATH" \
SINGULARITYENV_TRAINING_MIX_TRAINER_ARGV_JSON="$preflight_argv_json" \
singularity exec --nv \
  --bind /data/user/qinlucao:/data/user/qinlucao \
  "$TRAINER_IMAGE" "$TRAINER_ENV/bin/python" -c \
  'import json, os; from experiments.training_mix.train import main; raise SystemExit(main(json.loads(os.environ["TRAINING_MIX_TRAINER_ARGV_JSON"])))'
