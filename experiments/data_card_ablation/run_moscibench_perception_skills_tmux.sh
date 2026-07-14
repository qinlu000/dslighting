#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_ROOT="${MOSCIBENCH_DATA_ROOT:-$REPO_ROOT/data/releases/moscibench_full/moscibench}"
CONCURRENCY="${CONCURRENCY:-88}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
STDOUT_LOG="${STDOUT_LOG:-$LOG_DIR/moscibench_perception_skills_$RUN_STAMP.log}"
UV_BIN="${UV_BIN:-$REPO_ROOT/data/.uv/bin/uv}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="$REPO_ROOT/data/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/data/.uv/python"
export UV_TOOL_DIR="$REPO_ROOT/data/.uv/tools"
export UV_TOOL_BIN_DIR="$REPO_ROOT/data/.uv/bin"

if [[ ! -x "$UV_BIN" ]]; then
  echo "ERROR: uv is not executable: $UV_BIN" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: MoSciBench data root is missing: $DATA_ROOT" >&2
  exit 2
fi
for proxy_name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [[ -n "${!proxy_name-}" ]]; then
    echo "ERROR: proxy variable remains set: $proxy_name" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "[perception-skills] proxy variables unset"
echo "[perception-skills] data_root=$DATA_ROOT"
echo "[perception-skills] concurrency=$CONCURRENCY"
echo "[perception-skills] llm_max_concurrency=$CONCURRENCY"
echo "[perception-skills] llm_max_concurrent_per_key=$CONCURRENCY"
echo "[perception-skills] gpu_policy=cpu_default"
echo "[perception-skills] stdout_log=$STDOUT_LOG"
echo "[perception-skills] conditions=main,no-added-skill-v1,dataset-semantics-v1,scientific-modalities-v1"
echo "[perception-skills] git_head=$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "[perception-skills] git_worktree_dirty=true"
else
  echo "[perception-skills] git_worktree_dirty=false"
fi
echo "[perception-skills] runner_sha256=$(sha256sum experiments/data_card_ablation/run_perception_skills.py | awk '{print $1}')"

set +e
"$UV_BIN" run --no-sync python \
  experiments/data_card_ablation/run_perception_skills.py \
  --data-root "$DATA_ROOT" \
  --concurrency "$CONCURRENCY" \
  2>&1 | tee "$STDOUT_LOG"
exit_code="${PIPESTATUS[0]}"
set -e

echo "[perception-skills] exit_code=$exit_code"
exit "$exit_code"
