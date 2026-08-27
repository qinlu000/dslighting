#!/usr/bin/env bash
#SBATCH --job-name=dslight-ood-smoke
#SBATCH --partition=acd_ue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/ood-smoke-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
module load singularity/4.4.1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/data/user/qinlucao/envs/dslighting-agl-controller-tmix-v1}"
TRAINER_ENV="${TRAINER_ENV:-/data/user/qinlucao/envs/agent-lightning-trainer-cu130-tmix-v1}"
TRAINER_IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
DABENCH_ROOT="${DABENCH_ROOT:-/data/user/qinlucao/datasets/DABench/data}"
DABENCH_PYTHON="${DABENCH_PYTHON:-/data/user/qinlucao/envs/data-agent312-tmix-v1/bin/python}"
AGENTIC_ROOT="${AGENTIC_ROOT:-/data/user/qinlucao/datasets/AgenticDataBench}"
AGENTIC_PYTHON="${AGENTIC_PYTHON:-/data/user/qinlucao/envs/agenticdatabench310-tmix-v1/bin/python}"
: "${MODEL_DIR:?Pass the merged Hugging Face model directory as MODEL_DIR}"

OOD_FULL="${OOD_FULL:-0}"
OOD_CONCURRENCY="${OOD_CONCURRENCY:-1}"
if [[ "$OOD_FULL" == 1 ]]; then
  run_kind="ood-full"
else
  run_kind="ood-smoke"
fi
RUN_ROOT="${RUN_ROOT:-/data/user/qinlucao/runs/training_mix/${run_kind}-${SLURM_JOB_ID}}"
PORT="${VLLM_PORT:-$((24000 + SLURM_JOB_ID % 10000))}"
API_BASE="http://127.0.0.1:${PORT}/v1"
MODEL_NAME="dslighting-ood"
CACHE_ROOT="/tmp/dslighting-ood-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/logs" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/config" \
  "$CACHE_ROOT/torchinductor" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/cupy" \
  "$CACHE_ROOT/vllm" \
  "$CACHE_ROOT/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$CACHE_ROOT/config"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUPY_CACHE_DIR="$CACHE_ROOT/cupy"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export HF_HOME="$CACHE_ROOT/huggingface"
for name in TRITON_LIBCUDA_PATH XDG_CACHE_HOME XDG_CONFIG_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR CUPY_CACHE_DIR VLLM_CACHE_ROOT HF_HOME; do
  value="${!name}"
  export "SINGULARITYENV_${name}=$value"
done

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ "$CACHE_ROOT" == /tmp/dslighting-ood-* && -d "$CACHE_ROOT" ]]; then
    find "$CACHE_ROOT" -depth -delete 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
export SINGULARITYENV_DSLIGHTING_TRAINER_ENV="$TRAINER_ENV"
export SINGULARITYENV_DSLIGHTING_MODEL_DIR="$MODEL_DIR"
export SINGULARITYENV_DSLIGHTING_MODEL_NAME="$MODEL_NAME"
export SINGULARITYENV_DSLIGHTING_VLLM_PORT="$PORT"
singularity exec --nv --bind /data/user/qinlucao:/data/user/qinlucao \
  "$TRAINER_IMAGE" /bin/bash -c \
  'exec "$DSLIGHTING_TRAINER_ENV/bin/vllm" serve "$DSLIGHTING_MODEL_DIR" --served-model-name "$DSLIGHTING_MODEL_NAME" --host 127.0.0.1 --port "$DSLIGHTING_VLLM_PORT" --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85' \
  >"$RUN_ROOT/logs/vllm.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
  if "$CONTROLLER_ENV/bin/python" -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2).read()" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$RUN_ROOT/logs/vllm.log" >&2
    exit 2
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  tail -100 "$RUN_ROOT/logs/vllm.log" >&2
  exit 2
fi

evaluation_args=(
  --api-base "$API_BASE"
  --model "openai/$MODEL_NAME"
  --dabench-root "$DABENCH_ROOT"
  --dabench-python "$DABENCH_PYTHON"
  --agentic-root "$AGENTIC_ROOT"
  --agentic-python "$AGENTIC_PYTHON"
  --output-root "$RUN_ROOT/results"
  --max-steps 10
  --concurrency "$OOD_CONCURRENCY"
)
if [[ "$OOD_FULL" == 1 ]]; then
  evaluation_args+=(--full)
fi

"$CONTROLLER_ENV/bin/python" -m experiments.training_mix.hpc.ood_smoke \
  "${evaluation_args[@]}"
