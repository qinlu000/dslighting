#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
DATACOPE_ROOT=${DATACOPE_ROOT:-/data/caoqinlu/projects/DataMind/datacope/general}
DATACOPE_PY=${DATACOPE_PY:-/data/caoqinlu/projects/DataMind/.venv/bin/python}
RUNS=${RUNS:-"$ROOT/experiments/datacope_dabench/runs"}

run_sample() {
  local round=$1
  local sample=$2
  local skill=${3:-}
  local tag workspace predictions
  local -a args

  tag=$(printf '%02d' "$sample")
  workspace="$RUNS/workspaces/dabench_datacope_explore_round_${round}_sample_${tag}"
  predictions="$RUNS/predictions/round_${round}/run_${tag}"

  args=(
    -m experiments.datacope_dabench run
    --phase explore
    --round-index "$round"
    --sample-index "$sample"
  )
  if [ -n "$skill" ]; then
    args+=(--skill-file "$skill")
  fi
  "$PY" "${args[@]}"

  "$PY" -m experiments.datacope_dabench export \
    --workspace-root "$workspace" \
    --output-dir "$predictions"
}

run_round() {
  local round=$1
  local skill=${2:-}
  local sample
  for sample in $(seq 0 9); do
    run_sample "$round" "$sample" "$skill"
  done
}

discover_round() {
  local round=$1
  local previous_skill=${2:-}
  local -a history=()
  local -a args
  local previous

  for previous in $(seq 0 $((round - 1))); do
    history+=(--previous-predictions-dir "$RUNS/predictions/round_${previous}")
  done

  args=(
    -m experiments.datacope_dabench discover
    --python "$DATACOPE_PY"
    --datacope-root "$DATACOPE_ROOT"
    --round-index "$round"
    --predictions-dir "$RUNS/predictions/round_${round}"
    --verified-dir "$RUNS/verified/round_${round}"
    --data-dir "$RUNS/public_data"
    --skill-dir "$RUNS/skills/round_${round}/dabench"
    --model gpt-5.5
    "${history[@]}"
  )
  if [ -n "$previous_skill" ]; then
    args+=(--previous-skill-dir "$previous_skill")
  fi
  "$PY" "${args[@]}"
}

if [ "${RESUME_FROM_ROUND1:-0}" != "1" ] && [ "${RESUME_FROM_ROUND2:-0}" != "1" ]; then
  run_round 0

  "$PY" -m experiments.datacope_dabench prepare-data \
    --source-root "$ROOT/data/releases/dabench_perception_clean_v1" \
    --predictions-dir "$RUNS/predictions/round_0" \
    --output-dir "$RUNS/public_data"

  discover_round 0
fi

if [ "${RESUME_FROM_ROUND2:-0}" != "1" ]; then
  run_round 1 "$RUNS/skills/round_0/dabench/SKILL.md"
  discover_round 1 "$RUNS/skills/round_0/dabench"
fi

run_round 2 "$RUNS/skills/round_1/dabench/SKILL.md"
discover_round 2 "$RUNS/skills/round_1/dabench"

"$PY" -m experiments.datacope_dabench run --phase test

"$PY" -m experiments.datacope_dabench run \
  --phase test \
  --skill-file "$RUNS/skills/round_2/dabench/SKILL.md"

echo "DataCOPE × DABench experiment complete."
