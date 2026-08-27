#!/usr/bin/env bash
#SBATCH --job-name=tmix-sft-env
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix_sft/environment/setup-%j.out

set -euo pipefail
module load singularity/4.4.1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
WHEELHOUSE_ROOT="${WHEELHOUSE_ROOT:-/data/user/qinlucao/datasets/training_mix_hpc_wheelhouse_v1}"
SFT_WHEELHOUSE="${SFT_WHEELHOUSE:-/data/user/qinlucao/datasets/training_mix_sft_wheelhouse_v1}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/data/user/qinlucao/research/LLaMA-Factory-datamind-4b49ef0}"
IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
SFT_ENV="${SFT_ENV:-/data/user/qinlucao/envs/llamafactory-sft-tmix-v1}"
UV="${UV:-/data/user/qinlucao/bin/uv}"
PYTHON="$WHEELHOUSE_ROOT/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3"
REQUIREMENTS="$PROJECT_ROOT/experiments/training_mix/hpc/requirements_sft.lock.txt"
UV_CACHE_DIR="${UV_CACHE_DIR:-/data/user/qinlucao/cache/uv_sft}"

mkdir -p /data/user/qinlucao/runs/training_mix_sft/environment "$UV_CACHE_DIR"
for path in "$IMAGE" "$UV" "$PYTHON" "$REQUIREMENTS" "$LLAMAFACTORY_ROOT" \
  "$WHEELHOUSE_ROOT/trainer_cu128" "$SFT_WHEELHOUSE/SHA256SUMS"; do
  test -e "$path"
done
(cd "$SFT_WHEELHOUSE" && sha256sum --quiet -c SHA256SUMS)

container_exec() {
  local argv_json
  argv_json="$("$PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
  SINGULARITYENV_SFT_EXEC_ARGV_JSON="$argv_json" \
  SINGULARITYENV_UV_CACHE_DIR="$UV_CACHE_DIR" \
  SINGULARITYENV_UV_LINK_MODE=copy \
  singularity exec --cleanenv \
    --bind /data/user/qinlucao:/data/user/qinlucao \
    "$IMAGE" /usr/bin/python3 -c \
    'import json, os; argv=json.loads(os.environ["SFT_EXEC_ARGV_JSON"]); os.execvp(argv[0], argv)'
}

if [[ ! -x "$SFT_ENV/bin/python" ]]; then
  container_exec "$UV" venv --python "$PYTHON" "$SFT_ENV"
fi

container_exec "$UV" pip install --offline --no-index \
  --find-links "$WHEELHOUSE_ROOT/trainer_cu128" \
  --python "$SFT_ENV/bin/python" \
  'torch==2.8.0+cu128'

DS_BUILD_OPS=0 container_exec "$UV" pip install --offline --no-index --no-build-isolation \
  --find-links "$SFT_WHEELHOUSE" \
  --find-links "$WHEELHOUSE_ROOT/trainer_cu128" \
  --python "$SFT_ENV/bin/python" \
  -r "$REQUIREMENTS"

container_exec "$UV" pip install --offline --no-index --no-deps \
  --find-links "$WHEELHOUSE_ROOT/trainer_cu128" \
  --python "$SFT_ENV/bin/python" \
  'flash-attn==2.8.3'
container_exec "$UV" pip install --offline --no-index --no-deps --no-build-isolation --force-reinstall \
  --python "$SFT_ENV/bin/python" "$LLAMAFACTORY_ROOT"
container_exec "$UV" pip check --python "$SFT_ENV/bin/python"

container_exec "$SFT_ENV/bin/python" - <<'PY' \
  | tee "/data/user/qinlucao/runs/training_mix_sft/environment/setup-${SLURM_JOB_ID}.json"
import importlib
import importlib.metadata as metadata
import json
import platform
import torch

packages = [
    "llamafactory", "torch", "transformers", "datasets", "accelerate",
    "peft", "trl", "deepspeed", "flash-attn", "tensorboard",
]
for module in [
    "llamafactory", "llamafactory.data", "llamafactory.train.tuner",
    "transformers", "datasets", "accelerate", "peft", "trl", "deepspeed", "flash_attn",
]:
    importlib.import_module(module)
print(json.dumps({
    "node": platform.node(),
    "python": platform.python_version(),
    "packages": {name: metadata.version(name) for name in packages},
    "torch_cuda": torch.version.cuda,
}, indent=2))
PY

echo "environment_setup=passed end=$(date --iso-8601=seconds)"
