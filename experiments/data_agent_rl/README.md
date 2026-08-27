# Data Agent RL Environment via DSLighting

This adapter runs the deterministic subset of the local
`data_agent_rl_environment_train` dataset with a DSLighting workflow. It excludes
`flexible` and `llm_judge_long`, leaving 1,759 automatically graded tasks.

The controller downloads each referenced public HF Bucket prefix once into a
shared cache. The task sandbox receives only a read-only `input/` link, runs
without network access, and writes `answer.txt`. Gold answers remain outside the
agent workspace.

## Dry run

```bash
python -m experiments.data_agent_rl \
  --dataset-root /data/caoqinlu/datasets/data_agent_rl_environment_train \
  --output-dir runs/data-agent-rl-smoke \
  --index 0-5 \
  --workflow react \
  --model DeepSeek-V4-Flash \
  --provider openai \
  --sandbox-python /path/to/data-agent-rl/.venv/bin/python \
  --concurrency 5 \
  --compute-threads 2 \
  --dry-run
```

Remove `--dry-run` to execute. Existing successful artifacts are skipped;
`--retry-failed` reruns only tasks that did not create `answer.txt`.

Create the separate sandbox environment with:

```bash
uv venv /path/to/data-agent-rl/.venv --python 3.12.13
uv pip install \
  --python /path/to/data-agent-rl/.venv/bin/python \
  -r experiments/data_agent_rl/requirements_sandbox.txt
```
