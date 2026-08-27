#!/usr/bin/env bash
#SBATCH --job-name=dslight-env-setup
#SBATCH --partition=acd_ue
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/qinlucao/runs/training_mix/environment/setup-%j.out

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="${PROJECT_ROOT:-/data/user/qinlucao/research/dslighting-training-mix-20260826-final}"
WHEELHOUSE_ROOT="${WHEELHOUSE_ROOT:-/data/user/qinlucao/datasets/training_mix_hpc_wheelhouse_v1}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/data/user/qinlucao/envs/dslighting-agl-controller-tmix-v1}"
DARE_ENV="${DARE_ENV:-/data/user/qinlucao/envs/darebench310-tmix-v1}"
GENERAL_ENV="${GENERAL_ENV:-/data/user/qinlucao/envs/data-agent312-tmix-v1}"
AGENTIC_ENV="${AGENTIC_ENV:-/data/user/qinlucao/envs/agenticdatabench310-tmix-v1}"
UV="${UV:-/data/user/qinlucao/bin/uv}"
PY310="$WHEELHOUSE_ROOT/python/cpython-3.10.12-linux-x86_64-gnu/bin/python3"
PY312="$WHEELHOUSE_ROOT/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3"
AGL_WHEEL="$WHEELHOUSE_ROOT/sources/agentlightning-1.0.1-py3-none-any.whl"
DSLIGHTING_WHEEL="$WHEELHOUSE_ROOT/sources/dslighting-2.7.9-py3-none-any.whl"

for path in "$UV" "$PY310" "$PY312" "$WHEELHOUSE_ROOT/SHA256SUMS" \
  "$AGL_WHEEL" "$DSLIGHTING_WHEEL"; do
  test -e "$path"
done

(cd "$WHEELHOUSE_ROOT" && sha256sum --quiet -c SHA256SUMS)

mkdir -p /data/user/qinlucao/cache/uv /data/user/qinlucao/runs/training_mix/environment
export UV_CACHE_DIR=/data/user/qinlucao/cache/uv
export UV_LINK_MODE=hardlink

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date --iso-8601=seconds)"

install_host_environment() {
  local python="$1" environment="$2" wheels="$3" requirements="$4"
  if [[ ! -x "$environment/bin/python" ]]; then
    "$UV" venv --python "$python" "$environment"
  fi
  "$UV" pip install --offline --no-index --find-links "$wheels" \
    --python "$environment/bin/python" -r "$requirements"
  "$UV" pip check --python "$environment/bin/python"
}

install_host_environment \
  "$PY310" "$DARE_ENV" "$WHEELHOUSE_ROOT/dare" \
  "$PROJECT_ROOT/experiments/training_mix/hpc/requirements_dare_sandbox.lock.txt"
install_host_environment \
  "$PY312" "$GENERAL_ENV" "$WHEELHOUSE_ROOT/general" \
  "$PROJECT_ROOT/experiments/training_mix/hpc/requirements_general_sandbox.lock.txt"
install_host_environment \
  "$PY310" "$AGENTIC_ENV" "$WHEELHOUSE_ROOT/agentic" \
  "$PROJECT_ROOT/experiments/agenticdatabench/requirements_sandbox.lock.txt"
install_host_environment \
  "$PY312" "$CONTROLLER_ENV" "$WHEELHOUSE_ROOT/controller_manylinux228" \
  "$PROJECT_ROOT/experiments/training_mix/hpc/requirements_controller.lock.txt"
"$UV" pip install --offline --no-index --no-deps \
  --python "$CONTROLLER_ENV/bin/python" "$AGL_WHEEL" "$DSLIGHTING_WHEEL"
"$UV" pip check --python "$CONTROLLER_ENV/bin/python"

"$CONTROLLER_ENV/bin/python" - \
  "$DARE_ENV/bin/python" "$GENERAL_ENV/bin/python" "$AGENTIC_ENV/bin/python" \
  "$CONTROLLER_ENV/bin/python" \
  <<'PY' | tee "/data/user/qinlucao/runs/training_mix/environment/setup-${SLURM_JOB_ID}.json"
import json
import platform
import subprocess
import sys

def fingerprint(command, packages):
    program = """
import importlib, importlib.metadata as metadata, json, platform, sys
names=json.loads(sys.argv[1])
versions={name: metadata.version(name) for name in names}
modules={'scikit-learn': 'sklearn'}
for name in names:
    importlib.import_module(modules.get(name, name))
payload={'python': platform.python_version(), 'packages': versions}
print(json.dumps(payload))
"""
    return json.loads(subprocess.run(
        [*command, '-c', program, json.dumps(packages)],
        check=True, capture_output=True, text=True,
    ).stdout)

print(json.dumps({
  'node': platform.node(),
  'agent_lightning_commit': '8435586d147b4cf7bff33e687d7317149e79cbb8',
  'dare': fingerprint([sys.argv[1]], ['numpy', 'pandas', 'scikit-learn', 'pmdarima']),
  'general': fingerprint([sys.argv[2]], ['numpy', 'pandas', 'tensorflow', 'xgboost']),
  'agentic': fingerprint([sys.argv[3]], ['numpy', 'pandas', 'geopandas', 'duckdb']),
  'controller': fingerprint([sys.argv[4]], ['dslighting', 'agentlightning', 'pyarrow']),
}, indent=2))
PY

echo "environment_setup=passed end=$(date --iso-8601=seconds)"
