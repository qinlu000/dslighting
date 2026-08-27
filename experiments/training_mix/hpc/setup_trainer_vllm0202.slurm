#!/usr/bin/env bash
#SBATCH --job-name=tmix-vllm0202-env
#SBATCH --partition=acd_u
#SBATCH --nodelist=ACD1-[8,13,25,30]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/environment/vllm0202-setup-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
module load singularity/4.4.1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
WHEELHOUSE_ROOT="${WHEELHOUSE_ROOT:-/data/user/qinlucao/datasets/training_mix_hpc_wheelhouse_v1}"
TRAINER_IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
TRAINER_ENV="${TRAINER_ENV:-/data/user/qinlucao/envs/agent-lightning-trainer-cu130-tmix-v1}"
UV="${UV:-/data/user/qinlucao/bin/uv}"
PYTHON="$WHEELHOUSE_ROOT/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3"
TRAINER_WHEELS="$WHEELHOUSE_ROOT/trainer_glibc235"
TRAINER_LOCK="$PROJECT_ROOT/experiments/training_mix/hpc/requirements_trainer_vllm0202_cu130.lock.txt"
PATCHED_VERL="$WHEELHOUSE_ROOT/patches/verl_numpy2/patched/verl-0.8.0-py3-none-any.whl"
AGENT_LIGHTNING_WHEEL="$WHEELHOUSE_ROOT/sources/agentlightning-1.0.1-py3-none-any.whl"
FLASH_ATTN_SOURCE="$WHEELHOUSE_ROOT/sources/flash_attn-2.8.3.tar.gz"
FINGERPRINT_ROOT="/data/user/qinlucao/runs/training_mix/environment/vllm0202-cu130"

for path in "$PROJECT_ROOT" "$TRAINER_IMAGE" "$UV" "$PYTHON" \
  "$TRAINER_WHEELS" "$TRAINER_LOCK" "$PATCHED_VERL" \
  "$AGENT_LIGHTNING_WHEEL" "$FLASH_ATTN_SOURCE"; do
  test -e "$path"
done
command -v singularity >/dev/null
mkdir -p "$FINGERPRINT_ROOT" /data/user/qinlucao/cache/uv
cache_root="/tmp/tmix-vllm0202-env-${SLURM_JOB_ID}"
mkdir -p "$cache_root/cupy" "$cache_root/flashinfer" "$cache_root/xdg" \
  "$cache_root/huggingface" "$cache_root/vllm"
export CUPY_CACHE_DIR="$cache_root/cupy"
export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
export XDG_CACHE_HOME="$cache_root/xdg"
export HF_HOME="$cache_root/huggingface"
export VLLM_CACHE_ROOT="$cache_root/vllm"
for name in CUPY_CACHE_DIR FLASHINFER_WORKSPACE_BASE XDG_CACHE_HOME HF_HOME VLLM_CACHE_ROOT; do
  export "SINGULARITYENV_${name}=${!name}"
done

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
driver_major="${driver_version%%.*}"
if (( driver_major < 580 )); then
  echo "CUDA 13 trainer requires NVIDIA driver >=580; found $driver_version on $(hostname)" >&2
  exit 2
fi
echo "job=$SLURM_JOB_ID node=$(hostname) driver=$driver_version start=$(date --iso-8601=seconds)"

trainer_exec() {
  local argv_base64
  argv_base64="$("$PYTHON" -c \
    'import base64, json, sys; print(base64.b64encode(json.dumps(sys.argv[1:]).encode()).decode())' \
    "$@")"
  SINGULARITYENV_TRAINING_MIX_EXEC_ARGV_BASE64="$argv_base64" \
  singularity exec --nv --bind /data/user/qinlucao:/data/user/qinlucao \
    "$TRAINER_IMAGE" /usr/bin/python3 -c \
    'import base64, json, os; payload=base64.b64decode(os.environ["TRAINING_MIX_EXEC_ARGV_BASE64"]); argv=json.loads(payload); os.execvp(argv[0], argv)'
}

trainer_exec ldd --version
if [[ ! -x "$TRAINER_ENV/bin/python" ]]; then
  trainer_exec "$UV" venv --python "$PYTHON" "$TRAINER_ENV"
fi

export UV_CACHE_DIR=/data/user/qinlucao/cache/uv
export UV_LINK_MODE=hardlink
trainer_exec "$UV" pip install --offline --no-index \
  --find-links "$TRAINER_WHEELS" \
  --python "$TRAINER_ENV/bin/python" \
  -r "$TRAINER_LOCK"
trainer_exec "$UV" pip install --offline --no-index --no-deps \
  --python "$TRAINER_ENV/bin/python" \
  "$PATCHED_VERL" "$AGENT_LIGHTNING_WHEEL"

CUDA_HOME="$(trainer_exec "$TRAINER_ENV/bin/python" -c \
  'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"

if ! trainer_exec "$TRAINER_ENV/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export FLASH_ATTN_CUDA_ARCHS=90
  export MAX_JOBS="${MAX_JOBS:-12}"
  trainer_exec "$UV" pip install --offline --no-index --no-build-isolation \
    --no-deps --python "$TRAINER_ENV/bin/python" "$FLASH_ATTN_SOURCE"
fi
trainer_exec "$UV" pip check --python "$TRAINER_ENV/bin/python"

trainer_exec "$UV" pip freeze --python "$TRAINER_ENV/bin/python" \
  | tee "$FINGERPRINT_ROOT/requirements-${SLURM_JOB_ID}.lock.txt" >/dev/null
trainer_exec "$TRAINER_ENV/bin/python" - <<'PY' \
  | tee "$FINGERPRINT_ROOT/fingerprint-${SLURM_JOB_ID}.json"
import importlib
import importlib.metadata as metadata
import json

import cupy
import torch

expected = {
    "agentlightning": "1.0.1",
    "flash-attn": "2.8.3",
    "flashinfer-python": "0.6.8.post1",
    "torch": "2.11.0+cu130",
    "verl": "0.8.0",
    "vllm": "0.20.2",
}
versions = {name: metadata.version(name) for name in expected}
if versions != expected:
    raise SystemExit(f"trainer version mismatch: expected={expected}, actual={versions}")
for module in ("agentlightning", "flash_attn", "flashinfer", "verl", "vllm"):
    importlib.import_module(module)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("trainer CUDA preflight failed")
if cupy.cuda.runtime.getDeviceCount() != 1:
    raise SystemExit("CuPy CUDA preflight failed")
value = (torch.ones(32, device="cuda") * 2).sum().item()
if value != 64:
    raise SystemExit(f"unexpected CUDA result: {value}")
print(json.dumps({
    "versions": versions,
    "torch_cuda": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_result": value,
}, indent=2))
PY

echo "trainer_environment=passed end=$(date --iso-8601=seconds)"
