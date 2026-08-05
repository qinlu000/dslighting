#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
BENCHMARK_ROOT=${AGENTICDATABENCH_ROOT:?Set AGENTICDATABENCH_ROOT to the official checkout}
DATASET_ROOT=${AGENTICDATABENCH_DATASET_ROOT:-"$BENCHMARK_ROOT/testbed/datasets"}
DOCKER_IMAGE=${AGENTICDATABENCH_DOCKER_IMAGE:?Set AGENTICDATABENCH_DOCKER_IMAGE}
DATACOPE_ROOT=${DATACOPE_ROOT:?Set DATACOPE_ROOT to the DataCOPE checkout}
DATACOPE_PY=${DATACOPE_PY:?Set DATACOPE_PY to its Python executable}
EVALUATOR_PY=${EVALUATOR_PY:-"$PY"}
MODEL=${MODEL:-"openai/DeepSeek-V4-Flash"}
RUNS="$ROOT/experiments/datacope_agenticdatabench/runs"
SPLIT="$ROOT/experiments/datacope_agenticdatabench/split.json"

if [[ -e "$RUNS" ]]; then
  echo "Refusing to mix outputs; remove $RUNS before a fresh experiment." >&2
  exit 2
fi
if [[ ! -f "$SPLIT" ]]; then
  echo "Frozen split manifest is missing: $SPLIT" >&2
  exit 2
fi

cd "$ROOT"
"$PY" -m experiments.datacope_agenticdatabench prepare-data \
  --benchmark-root "$BENCHMARK_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --split-manifest "$SPLIT" \
  --output-dir "$RUNS/public_data"

for round in 0 1 2; do
  skill_args=()
  if (( round > 0 )); then
    skill_args=(--skill-file "$RUNS/skills/round_$((round - 1))/agenticdatabench/SKILL.md")
  fi
  for sample in 0 1 2; do
    sample_tag=$(printf '%02d' "$sample")
    run_name="agenticdatabench_datacope_explore_round_${round}_sample_${sample_tag}"
    "$PY" -m experiments.datacope_agenticdatabench run \
      --phase explore \
      --round-index "$round" \
      --sample-index "$sample" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --dataset-root "$DATASET_ROOT" \
      --split-manifest "$SPLIT" \
      --run-root "$RUNS" \
      --model "$MODEL" \
      --sandbox-backend docker \
      --docker-image "$DOCKER_IMAGE" \
      "${skill_args[@]}"
    "$PY" -m experiments.datacope_agenticdatabench export \
      --benchmark-root "$BENCHMARK_ROOT" \
      --run-output-dir "$RUNS/solver_outputs/$run_name" \
      --split-manifest "$SPLIT" \
      --output-dir "$RUNS/predictions/round_$round/run_$sample_tag"
  done

  discover_args=()
  for previous in $(seq 0 $((round - 1))); do
    discover_args+=(--previous-predictions-dir "$RUNS/predictions/round_$previous")
  done
  if (( round > 0 )); then
    discover_args+=(--previous-skill-dir "$RUNS/skills/round_$((round - 1))/agenticdatabench")
  fi
  "$PY" -m experiments.datacope_agenticdatabench discover \
    --python "$DATACOPE_PY" \
    --datacope-root "$DATACOPE_ROOT" \
    --round-index "$round" \
    --predictions-dir "$RUNS/predictions/round_$round" \
    --verified-dir "$RUNS/verified/round_$round" \
    --data-dir "$RUNS/public_data" \
    --skill-dir "$RUNS/skills/round_$round/agenticdatabench" \
    --split-manifest "$SPLIT" \
    --model gpt-5.5 \
    --timeout-seconds 7200 \
    "${discover_args[@]}"
done

"$PY" -m experiments.datacope_agenticdatabench run \
  --phase test \
  --benchmark-root "$BENCHMARK_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --split-manifest "$SPLIT" \
  --run-root "$RUNS" \
  --model "$MODEL" \
  --sandbox-backend docker \
  --docker-image "$DOCKER_IMAGE"
"$PY" -m experiments.datacope_agenticdatabench run \
  --phase test \
  --benchmark-root "$BENCHMARK_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --split-manifest "$SPLIT" \
  --run-root "$RUNS" \
  --model "$MODEL" \
  --sandbox-backend docker \
  --docker-image "$DOCKER_IMAGE" \
  --skill-file "$RUNS/skills/round_2/agenticdatabench/SKILL.md"

"$PY" -m experiments.datacope_agenticdatabench evaluate \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$RUNS/solver_outputs/agenticdatabench_datacope_test_baseline" \
  --python "$EVALUATOR_PY"
"$PY" -m experiments.datacope_agenticdatabench evaluate \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$RUNS/solver_outputs/agenticdatabench_datacope_test_skill" \
  --python "$EVALUATOR_PY"

echo "AgenticDataBench DataCOPE experiment complete: $RUNS"
