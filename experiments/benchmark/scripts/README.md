# ReAct benchmark runners

The shared entry point runs ReAct on every supported benchmark:

```bash
python experiments/benchmark/run_react_benchmark.py dabench
python experiments/benchmark/run_react_benchmark.py dacode
python experiments/benchmark/run_react_benchmark.py scienceagentbench
python experiments/benchmark/run_react_benchmark.py moscibench
```

It loads the repository `.env` by default. Set `DSLIGHTING_ENV_FILE` to use a
different file.

## Data roots

Each benchmark accepts an explicit local release path:

| Benchmark | Environment variable |
| --- | --- |
| DABench | `DSLIGHTING_DABENCH_DATA` |
| DACode | `DSLIGHTING_DACODE_DATA` |
| ScienceAgentBench | `DSLIGHTING_SCIENCEAGENTBENCH_DATA` |
| MosciBench | `DSLIGHTING_MOSCIBENCH_DATA` |

The runner fails before execution if the selected data root does not exist.

## Runtime controls

Common overrides are configured through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `openai/deepseek-ai/DeepSeek-V3.1-Terminus` | Model identifier; `LLM_MODEL` remains a fallback |
| `MAX_STEPS` | `10` | ReAct step budget |
| `PERCEPTION_ENABLED` | `false` | Enable FastPerception |
| `TASK_IDS` | all | Comma- or whitespace-separated task subset |
| `RUN_NAME` | generated | Stable run identifier |
| `BENCHMARK_LOG_DIR` | benchmark run directory | Result and debug-log root |
| `WORKSPACE_DIR` | framework default | Persistent workspace root |
| `KEEP_WORKSPACE` | `true` | Retain task workspaces |
| `SANDBOX_BACKEND` | `local` | `local`, `docker`, or another configured backend |
| `DOCKER_IMAGE` | none | Required when `SANDBOX_BACKEND=docker` |
| `DISABLE_NETWORK` | `false` | Disable sandbox networking |
| `SCHEDULER_POLICY` | `balanced` | Scheduler policy |
| `GPU_POLICY` | `auto` | GPU allocation policy |
| `MAX_CONCURRENCY` | `8` | Concurrent benchmark tasks |
| `LLM_MAX_CONCURRENCY` | `20` | Concurrent LLM requests |

For example, run two DABench tasks with FastPerception in Docker:

```bash
DSLIGHTING_DABENCH_DATA=/data/releases/dabench \
TASK_IDS="dabench-task-a,dabench-task-b" \
PERCEPTION_ENABLED=true \
SANDBOX_BACKEND=docker \
DOCKER_IMAGE=dslighting:latest \
python experiments/benchmark/run_react_benchmark.py dabench
```

## Repairing a result subset

`merge_subset_results.py` replaces rerun rows in a base CSV while preserving
the base row order. It validates the schema and exact task set, and refuses to
overwrite an existing output:

```bash
python experiments/benchmark/merge_subset_results.py \
  --base full-results.csv \
  --subset repaired-results.csv \
  --output merged-results.csv \
  --task-id task-a \
  --task-id task-b
```

## Local MosciBench debugging

Two wrappers remain for reproducing workspace and missing-submission issues:

```bash
bash experiments/benchmark/scripts/run_local_moscibench_dslighting_workspace_debug.sh
bash experiments/benchmark/scripts/run_local_moscibench_react_output_contract_debug.sh
```

Useful overrides include `SOURCE_MOSCIBENCH_DATA`, `PYTHON_BIN`,
`LOCAL_DEBUG_PROFILE`, `MOSCI_TASKS`, `MOSCI_TASK_PRESET`, `MAX_STEPS`, and
`KEEP_WORKSPACE`. Set `DRY_RUN=true` to validate selection and paths without
starting a benchmark.

Generated runs are kept under `experiments/benchmark/runs/`, which is ignored
by Git and remains local unless explicitly archived elsewhere.
