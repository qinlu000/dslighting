#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data/releases/moscibench_full/moscibench/competitions"}
DATACOPE_ROOT=${DATACOPE_ROOT:-"/data/caoqinlu/projects/DataMind/datacope/general"}
DATACOPE_PY=${DATACOPE_PY:-"/data/caoqinlu/projects/DataMind/.venv/bin/python"}
RUNS="$ROOT/experiments/datacope_moscibench/runs"
SPLIT="$ROOT/experiments/datacope_moscibench/split.json"

if [[ -e "$RUNS" || -e "$SPLIT" ]]; then
  echo "Refusing to mix outputs; remove $RUNS and $SPLIT before a fresh experiment." >&2
  exit 2
fi

cd "$ROOT"
"$PY" -m experiments.datacope_moscibench make-split \
  --data-root "$DATA_ROOT" \
  --output "$SPLIT"
"$PY" -m experiments.datacope_moscibench prepare-data \
  --data-root "$DATA_ROOT" \
  --split-manifest "$SPLIT" \
  --output-dir "$RUNS/public_data"

for round in 0 1 2; do
  skill_args=()
  if (( round > 0 )); then
    skill_args=(--skill-file "$RUNS/skills/round_$((round - 1))/moscibench/SKILL.md")
  fi
  for sample in $(seq 0 2); do
    sample_tag=$(printf '%02d' "$sample")
    "$PY" -m experiments.datacope_moscibench run \
      --phase explore \
      --round-index "$round" \
      --sample-index "$sample" \
      --data-root "$DATA_ROOT" \
      --split-manifest "$SPLIT" \
      --run-root "$RUNS" \
      "${skill_args[@]}"
    "$PY" -m experiments.datacope_moscibench export \
      --workspace-root "$RUNS/workspaces/moscibench_datacope_explore_round_${round}_sample_${sample_tag}" \
      --split-manifest "$SPLIT" \
      --output-dir "$RUNS/predictions/round_$round/run_$sample_tag"
  done

  discover_args=()
  for previous in $(seq 0 $((round - 1))); do
    discover_args+=(--previous-predictions-dir "$RUNS/predictions/round_$previous")
  done
  if (( round > 0 )); then
    discover_args+=(--previous-skill-dir "$RUNS/skills/round_$((round - 1))/moscibench")
  fi
  "$PY" -m experiments.datacope_moscibench discover \
    --python "$DATACOPE_PY" \
    --datacope-root "$DATACOPE_ROOT" \
    --round-index "$round" \
    --predictions-dir "$RUNS/predictions/round_$round" \
    --verified-dir "$RUNS/verified/round_$round" \
    --data-dir "$RUNS/public_data" \
    --skill-dir "$RUNS/skills/round_$round/moscibench" \
    --split-manifest "$SPLIT" \
    --model gpt-5.5 \
    "${discover_args[@]}"
done

"$PY" -m experiments.datacope_moscibench run \
  --phase test \
  --data-root "$DATA_ROOT" \
  --split-manifest "$SPLIT" \
  --run-root "$RUNS"
"$PY" -m experiments.datacope_moscibench run \
  --phase test \
  --data-root "$DATA_ROOT" \
  --split-manifest "$SPLIT" \
  --run-root "$RUNS" \
  --skill-file "$RUNS/skills/round_2/moscibench/SKILL.md"
"$PY" -m experiments.datacope_moscibench summarize --run-root "$RUNS"

echo "MoSciBench DataCOPE experiment complete: $RUNS"
