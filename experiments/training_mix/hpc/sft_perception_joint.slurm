#!/usr/bin/env bash
#SBATCH --job-name=tmix-perception-sft
#SBATCH --partition=acd_u
#SBATCH --exclude=ACD1-63
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix_sft/perception-%j.out

set -euo pipefail
module load singularity/4.4.1

LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/data/user/qinlucao/research/LLaMA-Factory-datamind-4b49ef0}"
IMAGE="${TRAINER_IMAGE:-/data/user/qinlucao/software/containers/hpc-benchmarks_24.09.sif}"
SFT_ENV="${SFT_ENV:-/data/user/qinlucao/envs/llamafactory-sft-tmix-v1}"
MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
SFT_CONFIG="${SFT_CONFIG:?SFT_CONFIG is required}"
EXPECTED_RECORDS="${EXPECTED_RECORDS:-511}"
EXPECTED_SOLVER_RECORDS="${EXPECTED_SOLVER_RECORDS:-276}"
EXPECTED_PERCEPTION_RECORDS="${EXPECTED_PERCEPTION_RECORDS:-235}"
EXPECTED_TRAIN_SHA256="${EXPECTED_TRAIN_SHA256:?EXPECTED_TRAIN_SHA256 is required}"
ARGV_PYTHON="${ARGV_PYTHON:-/data/user/qinlucao/datasets/training_mix_hpc_wheelhouse_v1/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3}"

for path in "$IMAGE" "$SFT_ENV/bin/llamafactory-cli" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/model.safetensors.index.json" "$MODEL_ROOT/MODEL_SHA256SUMS" \
  "$DATA_ROOT/dataset_info.json" "$DATA_ROOT/train.json" \
  "$LLAMAFACTORY_ROOT/examples/deepspeed/ds_z2_config.json" \
  "$LLAMAFACTORY_ROOT/examples/deepspeed/ds_z3_config.json" "$SFT_CONFIG"; do
  test -e "$path"
done
(cd "$MODEL_ROOT" && sha256sum --quiet -c MODEL_SHA256SUMS)
echo "$EXPECTED_TRAIN_SHA256  $DATA_ROOT/train.json" | sha256sum --check --status

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
test "$gpu_count" -eq 4
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOB_CACHE="/tmp/qinlucao-sft-${SLURM_JOB_ID}"
mkdir -p \
  "$JOB_CACHE/hf" "$JOB_CACHE/torch" "$JOB_CACHE/datasets" \
  "$JOB_CACHE/triton" "$JOB_CACHE/matplotlib" "$JOB_CACHE/xdg"
export HF_HOME="$JOB_CACHE/hf"
export TORCH_EXTENSIONS_DIR="$JOB_CACHE/torch"
export HF_DATASETS_CACHE="$JOB_CACHE/datasets"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
export MPLCONFIGDIR="$JOB_CACHE/matplotlib"
export XDG_CACHE_HOME="$JOB_CACHE/xdg"

container_exec() {
  local argv_json
  argv_json="$("$ARGV_PYTHON" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"
  SINGULARITYENV_SFT_EXEC_ARGV_JSON="$argv_json" \
  SINGULARITYENV_PREPEND_PATH="$SFT_ENV/bin" \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  SINGULARITYENV_HF_HUB_OFFLINE=1 \
  SINGULARITYENV_TRANSFORMERS_OFFLINE=1 \
  SINGULARITYENV_TOKENIZERS_PARALLELISM=false \
  SINGULARITYENV_PYTHONUNBUFFERED=1 \
  SINGULARITYENV_OMP_NUM_THREADS=8 \
  SINGULARITYENV_FORCE_TORCHRUN=1 \
  SINGULARITYENV_NNODES=1 \
  SINGULARITYENV_NODE_RANK=0 \
  SINGULARITYENV_NPROC_PER_NODE=4 \
  SINGULARITYENV_MASTER_ADDR="$MASTER_ADDR" \
  SINGULARITYENV_MASTER_PORT="$MASTER_PORT" \
  SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SINGULARITYENV_HF_HOME="$HF_HOME" \
  SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  SINGULARITYENV_MPLCONFIGDIR="$MPLCONFIGDIR" \
  SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  singularity exec --nv --cleanenv \
    --bind /data/user/qinlucao:/data/user/qinlucao \
    --bind "$JOB_CACHE:$JOB_CACHE" \
    "$IMAGE" /usr/bin/python3 -c \
    'import json, os; argv=json.loads(os.environ["SFT_EXEC_ARGV_JSON"]); os.execvp(argv[0], argv)'
}

container_exec "$SFT_ENV/bin/python" - \
  "$MODEL_ROOT" "$DATA_ROOT" "$SFT_CONFIG" \
  "$EXPECTED_RECORDS" "$EXPECTED_SOLVER_RECORDS" "$EXPECTED_PERCEPTION_RECORDS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

model_root, data_root, config_path = map(Path, sys.argv[1:4])
expected_records, expected_solver, expected_perception = map(int, sys.argv[4:7])
config = yaml.safe_load(config_path.read_text())
assert Path(config["model_name_or_path"]) == model_root
assert Path(config["dataset_dir"]) == data_root
assert config["dataset"] == "training_mix_perception_joint_v1"
assert config["enable_thinking"] is False
assert config["cutoff_len"] == 24576
assert config["save_strategy"] == "no"
assert not Path(config["output_dir"]).exists(), config["output_dir"]

rows = json.loads((data_root / "train.json").read_text())
assert len(rows) == expected_records, len(rows)
roles = Counter(row["metadata"]["role"] for row in rows)
assert roles == {"solver": expected_solver, "perception": expected_perception}, roles
tokenizer = AutoTokenizer.from_pretrained(
    model_root,
    local_files_only=True,
    trust_remote_code=True,
)
lengths = [
    len(
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    for row in rows
]
assert max(lengths) <= 24576, max(lengths)
print(json.dumps({
    "examples": len(rows),
    "roles": roles,
    "min_tokens": min(lengths),
    "max_tokens": max(lengths),
    "output_dir": config["output_dir"],
}))
PY

echo "job=$SLURM_JOB_ID node=$(hostname) config=$SFT_CONFIG start=$(date --iso-8601=seconds)"
container_exec "$SFT_ENV/bin/llamafactory-cli" train "$SFT_CONFIG"

container_exec "$SFT_ENV/bin/python" - "$SFT_CONFIG" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
output_dir = Path(config["output_dir"])
state = json.loads((output_dir / "trainer_state.json").read_text())
results = json.loads((output_dir / "train_results.json").read_text())
assert state["global_step"] > 0, state["global_step"]
assert math.isfinite(float(results["train_loss"])), results["train_loss"]
assert (output_dir / "model.safetensors.index.json").is_file()
assert any((output_dir / "tensorboard").glob("events.out.tfevents.*"))
print(json.dumps({
    "output_dir": str(output_dir),
    "global_step": state["global_step"],
    "train_loss": results["train_loss"],
    "artifacts_valid": True,
}))
PY

echo "training=passed end=$(date --iso-8601=seconds)"
