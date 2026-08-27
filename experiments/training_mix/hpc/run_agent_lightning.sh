#!/usr/bin/env bash
set -euo pipefail

trainer_exec_mode="${TRAINER_EXEC_MODE:-container}"
if [[ "$trainer_exec_mode" != "container" && "$trainer_exec_mode" != "native" ]]; then
  echo "TRAINER_EXEC_MODE must be container or native" >&2
  exit 2
fi

required=(
  PROJECT_ROOT BUNDLE_ROOT CONTROLLER_ENV TRAINER_ENV DARE_PYTHON GENERAL_PYTHON
  MODEL_PATH RUN_ROOT
)
if [[ "$trainer_exec_mode" == "container" ]]; then
  required+=(TRAINER_IMAGE)
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
done
if [[ "$trainer_exec_mode" == "container" ]]; then
  command -v singularity >/dev/null
fi

if [[ "${MIN_NVIDIA_DRIVER_MAJOR:-0}" =~ ^[0-9]+$ ]] \
  && (( MIN_NVIDIA_DRIVER_MAJOR > 0 )); then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  driver_major="${driver_version%%.*}"
  if (( driver_major < MIN_NVIDIA_DRIVER_MAJOR )); then
    echo "NVIDIA driver $driver_version is older than required major $MIN_NVIDIA_DRIVER_MAJOR" >&2
    exit 2
  fi
fi

controller_python="$CONTROLLER_ENV/bin/python"
trainer_python="$TRAINER_ENV/bin/python"
server="$CONTROLLER_ENV/bin/agl-server"
controller="$CONTROLLER_ENV/bin/agl-controller"
manifest="$BUNDLE_ROOT/manifests/train_full.jsonl"

runtime_paths=(
  "$controller_python" "$trainer_python" "$server" "$controller"
  "$DARE_PYTHON" "$GENERAL_PYTHON" "$manifest"
)
if [[ "$trainer_exec_mode" == "container" ]]; then
  runtime_paths+=("$TRAINER_IMAGE")
fi
for path in "${runtime_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file does not exist: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "MODEL_PATH must be a local model directory: $MODEL_PATH" >&2
  exit 2
fi

rollout_concurrency="${ROLLOUT_CONCURRENCY:-16}"
compute_threads="${TRAINING_COMPUTE_THREADS:-2}"
max_steps="${TRAINING_MAX_STEPS:-10}"
timeout_seconds="${TRAINING_TIMEOUT_SECONDS:-600}"
rollout_timeout_seconds="${TRAINING_ROLLOUT_TIMEOUT_SECONDS:-1800}"
agent_deadline_seconds="${TRAINING_AGENT_DEADLINE_SECONDS:-1650}"
format_reward_weight="${TRAINING_FORMAT_REWARD_WEIGHT:-0.1}"
perception_enabled="${TRAINING_PERCEPTION_ENABLED:-0}"
gpus="${TRAINING_GPUS:-8}"
train_batch_size="${TRAIN_BATCH_SIZE:-40}"
group_size="${GROUP_SIZE:-4}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
ray_num_cpus="${RAY_NUM_CPUS:-$((gpus * 4))}"
total_epochs="${TOTAL_EPOCHS:-1}"
save_freq="${SAVE_FREQ:-10}"
run_name="${RUN_NAME:-training_mix_grpo}"
agl_key="${AGL_KEY:-dslighting-local}"
if [[ -z "${AGL_SERVER_PORT:-}" ]]; then
  if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    agl_server_port=$((20000 + SLURM_JOB_ID % 10000))
  else
    agl_server_port=8181
  fi
else
  agl_server_port="$AGL_SERVER_PORT"
fi
agl_base_url="http://127.0.0.1:$agl_server_port"

if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && (( ray_num_cpus > SLURM_CPUS_PER_TASK )); then
    echo "RAY_NUM_CPUS=$ray_num_cpus exceeds SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" >&2
    exit 2
fi
if ! [[ "$rollout_timeout_seconds" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$agent_deadline_seconds" =~ ^[1-9][0-9]*$ ]] \
  || (( agent_deadline_seconds >= rollout_timeout_seconds )); then
  echo "TRAINING_AGENT_DEADLINE_SECONDS must be positive and smaller than TRAINING_ROLLOUT_TIMEOUT_SECONDS" >&2
  exit 2
fi
if [[ "$perception_enabled" != 0 && "$perception_enabled" != 1 ]]; then
  echo "TRAINING_PERCEPTION_ENABLED must be 0 or 1" >&2
  exit 2
fi

cache_root="/tmp/dslighting-training-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_ROOT/logs" \
  "$RUN_ROOT/scratch" \
  "$cache_root/xdg" \
  "$cache_root/config" \
  "$cache_root/torchinductor" \
  "$cache_root/triton" \
  "$cache_root/cupy" \
  "$cache_root/flashinfer" \
  "$cache_root/vllm" \
  "$cache_root/huggingface"
checkpoint_dir="$RUN_ROOT/checkpoints"
mkdir -p "$checkpoint_dir"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Ray's object-store socket must stay below the Linux AF_UNIX path limit.
export RAY_TMPDIR="/tmp/dsl-ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
export TRAINING_SCRATCH_ROOT="$RUN_ROOT/scratch"
export TRAINING_ROLLOUT_AUDIT_PATH="$RUN_ROOT/rollout_audit.jsonl"
export TRAINING_MIX_MANIFEST="$manifest"
export DARE_MANIFEST="$BUNDLE_ROOT/dare/tasks.jsonl"
export DARE_DATABASES_DIR="$BUNDLE_ROOT/dare/databases"
export DATA_AGENT_ROOT="$BUNDLE_ROOT/data_agent/dataset"
export DATA_AGENT_CACHE="$BUNDLE_ROOT/data_agent/cache"
export DATAMIND_ROOT="$BUNDLE_ROOT/datamind"
export DARE_PYTHON GENERAL_PYTHON
export TRAINING_COMPUTE_THREADS="$compute_threads"
export TRAINING_MAX_STEPS="$max_steps"
export TRAINING_TIMEOUT_SECONDS="$timeout_seconds"
export TRAINING_ROLLOUT_TIMEOUT_SECONDS="$rollout_timeout_seconds"
export TRAINING_AGENT_DEADLINE_SECONDS="$agent_deadline_seconds"
export TRAINING_FORMAT_REWARD_WEIGHT="$format_reward_weight"
export TRAINING_PERCEPTION_ENABLED="$perception_enabled"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
# Some Slurm nodes export CUDA and ROCm visibility variables together. VERL
# deliberately rejects that ambiguous state; this workload is NVIDIA-only.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
# Keep the Slurm-visible GPU list intact and let VERL select the Ray-assigned
# local device. Ray's fractional colocated actors can otherwise both expose
# physical GPU 0 even when their accelerator IDs are 0 and 1.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
if [[ "$trainer_exec_mode" == "container" ]]; then
  # Singularity injects the NVIDIA driver into this directory, while the
  # image's stale ldconfig cache points Triton at a missing compat directory.
  export TRITON_LIBCUDA_PATH=/.singularity.d/libs
else
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"
fi
export XDG_CACHE_HOME="$cache_root/xdg"
export XDG_CONFIG_HOME="$cache_root/config"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TRITON_CACHE_DIR="$cache_root/triton"
export CUPY_CACHE_DIR="$cache_root/cupy"
export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export HF_HOME="$cache_root/huggingface"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_RAY_TMPDIR="$RAY_TMPDIR"
export SINGULARITYENV_TRAINING_SCRATCH_ROOT="$TRAINING_SCRATCH_ROOT"
export SINGULARITYENV_TRAINING_ROLLOUT_AUDIT_PATH="$TRAINING_ROLLOUT_AUDIT_PATH"
export SINGULARITYENV_TRAINING_MIX_MANIFEST="$TRAINING_MIX_MANIFEST"
for name in TRITON_LIBCUDA_PATH XDG_CACHE_HOME XDG_CONFIG_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR CUPY_CACHE_DIR FLASHINFER_WORKSPACE_BASE VLLM_CACHE_ROOT HF_HOME; do
  value="${!name}"
  export "SINGULARITYENV_${name}=$value"
done

server_pid=""
controller_pid=""
cleanup() {
  if [[ -n "$controller_pid" ]]; then
    kill "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ "$RAY_TMPDIR" == /tmp/dsl-ray-* && -d "$RAY_TMPDIR" ]]; then
    find "$RAY_TMPDIR" -depth -delete 2>/dev/null || true
  fi
  if [[ "$cache_root" == /tmp/dslighting-training-* && -d "$cache_root" ]]; then
    find "$cache_root" -depth -delete 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
trainer_exec() {
  if [[ "$trainer_exec_mode" == "native" ]]; then
    "$@"
  else
    singularity exec --nv --bind /data/user/qinlucao:/data/user/qinlucao \
      "$TRAINER_IMAGE" "$@"
  fi
}

trainer_exec "$trainer_python" - "$gpus" <<'PY'
import sys
from pathlib import Path
import cupy
import torch
from triton.backends.nvidia.driver import libcuda_dirs

expected = int(sys.argv[1])
available = torch.cuda.device_count()
if not torch.cuda.is_available() or available != expected:
    raise SystemExit(
        f"CUDA preflight failed: available={torch.cuda.is_available()} "
        f"devices={available} expected={expected} torch={torch.__version__}"
    )
if cupy.cuda.runtime.getDeviceCount() != expected:
    raise SystemExit("CuPy CUDA preflight did not see the Slurm GPU allocation")
driver_dirs = libcuda_dirs()
if not any((Path(path) / "libcuda.so.1").is_file() for path in driver_dirs):
    raise SystemExit(f"Triton CUDA driver preflight failed: {driver_dirs}")
print(
    f"cuda_preflight=passed devices={available} "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"cupy={cupy.__version__} triton_libcuda={driver_dirs}"
)
PY

"$server" \
  host=127.0.0.1 \
  port="$agl_server_port" \
  key="$agl_key" \
  default_proxy.model_name="$MODEL_PATH" \
  default_proxy.include_log_probs=true \
  >"$RUN_ROOT/logs/agl-server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if "$controller_python" -c \
    "import urllib.request; urllib.request.urlopen('$agl_base_url/healthz', timeout=2).read()" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -100 "$RUN_ROOT/logs/agl-server.log" >&2
    exit 2
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  tail -100 "$RUN_ROOT/logs/agl-server.log" >&2
  echo "Agent Lightning server did not become healthy" >&2
  exit 2
fi

"$controller" \
  runner_type=local \
  agl_server.url="$agl_base_url" \
  agl_server.key="$agl_key" \
  local_runner.maximum_size="$rollout_concurrency" \
  local_runner.poll_interval=1 \
  >"$RUN_ROOT/logs/agl-controller.log" 2>&1 &
controller_pid=$!

trainer_args=(
  --manifest "$manifest"
  --model "$MODEL_PATH"
  --output-dir "$checkpoint_dir"
  --gpus "$gpus"
  --train-batch-size "$train_batch_size"
  --group-size "$group_size"
  --tensor-parallel-size "$tensor_parallel_size"
  --ray-num-cpus "$ray_num_cpus"
  --total-epochs "$total_epochs"
  --save-freq "$save_freq"
  --agl-base-url "$agl_base_url"
  --agl-key "$agl_key"
  --run-name "$run_name"
  --rollout-timeout-seconds "$rollout_timeout_seconds"
)
if [[ -n "${LIMIT_TASKS:-}" ]]; then
  trainer_args+=(--limit-tasks "$LIMIT_TASKS")
fi

# HPC3's container wrapper treats any existing directory in the host command
# line as a sandbox image. Pass trainer arguments through the environment so
# model and output directories are only expanded inside the container.
trainer_argv_json="$("$controller_python" -c \
  'import json, sys; print(json.dumps(sys.argv[1:]))' "${trainer_args[@]}")"
export TRAINING_MIX_TRAINER_ARGV_JSON="$trainer_argv_json"
export SINGULARITYENV_TRAINING_MIX_TRAINER_ARGV_JSON="$trainer_argv_json"
trainer_exec "$trainer_python" -c \
  'import json, os; from experiments.training_mix.train import main; raise SystemExit(main(json.loads(os.environ["TRAINING_MIX_TRAINER_ARGV_JSON"])))' \
  2>&1 | tee "$RUN_ROOT/logs/trainer.log"

if [[ ! -f "$checkpoint_dir/latest_checkpointed_iteration.txt" ]]; then
  echo "Training exited without a VERL checkpoint: $checkpoint_dir" >&2
  exit 2
fi
