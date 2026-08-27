#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/caoqinlu/projects/dslighting}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/data/caoqinlu/runs/training_mix_perception_cold_start/rerun}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/paper/training_mix/artifacts/cold_start/perception_sft_rerun_qwen3_24k}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-$PROJECT_ROOT/paper/training_mix/artifacts/cold_start/teacher_candidates_900.jsonl}"
TASK_MANIFEST="${TASK_MANIFEST:-$PROJECT_ROOT/paper/training_mix/artifacts/manifests/training_mix_2000.jsonl}"
DARE_MANIFEST="${DARE_MANIFEST:-$PROJECT_ROOT/paper/training_mix/artifacts/manifests/dare_bench_600.jsonl}"
DARE_DATABASES_DIR="${DARE_DATABASES_DIR:-/data/caoqinlu/datasets/dare-bench/huggingface/train/databases}"
DATA_AGENT_ROOT="${DATA_AGENT_ROOT:-/data/caoqinlu/datasets/data_agent_rl_environment_train}"
DATA_AGENT_CACHE="${DATA_AGENT_CACHE:-/data/caoqinlu/datasets/data_agent_rl_environment_train_bucket_cache}"
DATAMIND_ROOT="${DATAMIND_ROOT:-/data/caoqinlu/datasets/DataMind-Data}"
DARE_PYTHON="${DARE_PYTHON:-/data/caoqinlu/datasets/dare-bench/repo/.venv/bin/python}"
GENERAL_PYTHON="${GENERAL_PYTHON:-/data/caoqinlu/datasets/data_agent_rl_environment_train/.venv/bin/python}"
TOKENIZER="${TOKENIZER:-/data/caoqinlu/models/Qwen3-4B-Instruct-2507}"
DARE_CONCURRENCY="${DARE_CONCURRENCY:-16}"
DATA_AGENT_CONCURRENCY="${DATA_AGENT_CONCURRENCY:-16}"
DATAMIND_CONCURRENCY="${DATAMIND_CONCURRENCY:-16}"
COMPUTE_THREADS="${COMPUTE_THREADS:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
TEMPERATURE="${TEMPERATURE:-0.9}"

dotenv_value() {
  "$PYTHON" - "$PROJECT_ROOT/.env" "$1" <<'PY'
import sys
from dotenv import dotenv_values

print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or "")
PY
}
MODEL="${MODEL:-$(dotenv_value LLM_MODEL)}"
MODEL="${MODEL:-DeepSeek-V4-Flash}"
DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-$(dotenv_value API_BASE)}"
if [[ -z "$DEEPSEEK_API_BASE" || -z "$(dotenv_value API_KEY)" ]]; then
  echo "API_BASE and API_KEY must be set in the environment or .env" >&2
  exit 2
fi

for path in "$PYTHON" "$SELECTION_MANIFEST" "$TASK_MANIFEST" "$DARE_MANIFEST" \
  "$DARE_DATABASES_DIR" "$DATA_AGENT_ROOT" "$DATA_AGENT_CACHE" "$DATAMIND_ROOT" \
  "$DARE_PYTHON" "$GENERAL_PYTHON" "$TOKENIZER"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done
for value in "$DARE_CONCURRENCY" "$DATA_AGENT_CONCURRENCY" "$DATAMIND_CONCURRENCY" \
  "$COMPUTE_THREADS" "$MAX_STEPS" "$TIMEOUT_SECONDS"; do
  if (( value <= 0 )); then
    echo "Concurrency, threads, steps, and timeout values must be positive" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/logs" "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$COMPUTE_THREADS"
export MKL_NUM_THREADS="$COMPUTE_THREADS"
export OPENBLAS_NUM_THREADS="$COMPUTE_THREADS"

common_args=(
  --manifest "$TASK_MANIFEST"
  --selection-manifest "$SELECTION_MANIFEST"
  --dare-manifest "$DARE_MANIFEST"
  --dare-databases-dir "$DARE_DATABASES_DIR"
  --data-agent-root "$DATA_AGENT_ROOT"
  --data-agent-cache "$DATA_AGENT_CACHE"
  --datamind-root "$DATAMIND_ROOT"
  --dare-python "$DARE_PYTHON"
  --general-python "$GENERAL_PYTHON"
  --run-root "$RUN_ROOT"
  --model "$MODEL"
  --api-base "$DEEPSEEK_API_BASE"
  --temperature "$TEMPERATURE"
  --compute-threads "$COMPUTE_THREADS"
  --max-steps "$MAX_STEPS"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --protocol-mode strict_retry
  --perception
  --repetition 0
)

pids=()
sources=(dare_bench data_agent_rl datamind_sql)
for source in "${sources[@]}"; do
  case "$source" in
    dare_bench) concurrency="$DARE_CONCURRENCY" ;;
    data_agent_rl) concurrency="$DATA_AGENT_CONCURRENCY" ;;
    datamind_sql) concurrency="$DATAMIND_CONCURRENCY" ;;
  esac
  "$PYTHON" -m experiments.training_mix.rollout run-source \
    --source "$source" \
    --concurrency "$concurrency" \
    "${common_args[@]}" \
    >"$RUN_ROOT/logs/$source.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!sources[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Teacher rollout failed: ${sources[$index]} (see $RUN_ROOT/logs/${sources[$index]}.log)" >&2
    failed=1
  fi
done
if [[ "$failed" == 1 ]]; then
  exit 1
fi

"$PYTHON" -m paper.training_mix.code.export_perception_cold_start_sft \
  --run-root "$RUN_ROOT" \
  --candidates "$SELECTION_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --cutoff-len 24576 \
  --seed 42 \
  >"$RUN_ROOT/logs/export.log" 2>&1

echo "Perception SFT data ready: $OUTPUT_DIR/llamafactory"
