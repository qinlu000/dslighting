#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-"$ROOT/.venv/bin/python"}
DATA_ROOT=${DATA_ROOT:-"$ROOT/data/releases/moscibench_full/moscibench/competitions"}
DATACOPE_ROOT=${DATACOPE_ROOT:-"/data/caoqinlu/projects/DataMind/datacope/general"}
DATACOPE_PY=${DATACOPE_PY:-"/data/caoqinlu/projects/DataMind/.venv/bin/python"}
RUNS="$ROOT/experiments/datacope_moscibench/runs"
SPLIT="$ROOT/experiments/datacope_moscibench/split.json"

cd "$ROOT"

check_codex_network() {
  curl --no-alpn --connect-timeout 5 --max-time 15 --silent --show-error \
    --output /dev/null https://chatgpt.com || {
    echo "Codex network unavailable; reconnect Traffic forwarding before discovery." >&2
    return 1
  }
}

check_codex_network
"$PY" -m experiments.datacope_moscibench discover \
  --python "$DATACOPE_PY" \
  --datacope-root "$DATACOPE_ROOT" \
  --round-index 0 \
  --predictions-dir "$RUNS/predictions/round_0" \
  --verified-dir "$RUNS/verified/round_0" \
  --data-dir "$RUNS/public_data" \
  --skill-dir "$RUNS/skills/round_0/moscibench" \
  --split-manifest "$SPLIT" \
  --model gpt-5.5 \
  --timeout-seconds 7200

for round in 1 2; do
  skill="$RUNS/skills/round_$((round - 1))/moscibench/SKILL.md"
  for sample in $(seq 0 2); do
    sample_tag=$(printf '%02d' "$sample")
    "$PY" -m experiments.datacope_moscibench run \
      --phase explore \
      --round-index "$round" \
      --sample-index "$sample" \
      --data-root "$DATA_ROOT" \
      --split-manifest "$SPLIT" \
      --run-root "$RUNS" \
      --skill-file "$skill"
    "$PY" -m experiments.datacope_moscibench export \
      --workspace-root "$RUNS/workspaces/moscibench_datacope_explore_round_${round}_sample_${sample_tag}" \
      --split-manifest "$SPLIT" \
      --output-dir "$RUNS/predictions/round_$round/run_$sample_tag"
  done

  previous=()
  for index in $(seq 0 $((round - 1))); do
    previous+=(--previous-predictions-dir "$RUNS/predictions/round_$index")
  done
  check_codex_network
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
    --previous-skill-dir "$RUNS/skills/round_$((round - 1))/moscibench" \
    --timeout-seconds 7200 \
    "${previous[@]}"
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
