# DARE-Bench via DSLighting workflows

This adapter keeps DARE-Bench task construction and deterministic reward while
using a DSLighting workflow as the solving agent. It targets the 2,818 training
samples in `train_fixed_gt.jsonl`; classification/regression v1 samples that
require regenerated references are intentionally excluded.

The agent receives only the files declared in `agent_payload.needed_files`.
Ground truth, metadata used by the grader, and the manifest are kept outside the
agent workspace. The adapter reads directly from `databases.zip`, so full
extraction is optional.

## Validate the full run plan

```bash
python -m experiments.dare_bench \
  --manifest /path/to/derived/fixed_gt_train/train_fixed_gt.jsonl \
  --databases-zip /path/to/train/databases.zip \
  --output-dir runs/dare-v4flash \
  --all \
  --workflow react \
  --model DeepSeek-V4-Flash \
  --provider openai \
  --sandbox-python /path/to/dare-bench/.venv/bin/python \
  --concurrency 60 \
  --dry-run
```

## Four-category smoke run

Select one exact task ID from each of classification v2, regression v2, time
series v1, and time series v2, and repeat `--task` four times:

```bash
python -m experiments.dare_bench \
  --manifest /path/to/derived/fixed_gt_train/train_fixed_gt.jsonl \
  --databases-zip /path/to/train/databases.zip \
  --output-dir runs/dare-v4flash-smoke \
  --task 'classification-base-id::v2' \
  --task 'regression-base-id::v2' \
  --task 'time-series-base-id::v1' \
  --task 'time-series-base-id::v2' \
  --workflow react \
  --model DeepSeek-V4-Flash \
  --provider openai \
  --sandbox-python /path/to/dare-bench/.venv/bin/python \
  --concurrency 4 \
  --max-steps 10
```

The default runtime is local Bubblewrap with sandbox network disabled; Docker is
not required. `--sandbox-python` keeps the controller environment separate from
the Python environment used for agent code. Each task writes `prediction.csv` and `result.json`. The run root
contains `run_summary.json`, including per-task reward, average reward, cost,
and status. Solver trajectories remain in the configured workspace directory.
