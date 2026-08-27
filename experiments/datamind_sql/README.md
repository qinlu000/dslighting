# DataMind SQL

This adapter runs the deterministic SQL subset selected in
`paper/training_mix`. The agent sees only a read-only `database.sqlite` and
must write `result.csv`. Ground-truth CSV files remain outside the sandbox and
are scored with DataMind's programmatic, order-insensitive table matcher.

```bash
.venv/bin/python -m experiments.datamind_sql \
  --manifest paper/training_mix/artifacts/manifests/training_mix_2000.jsonl \
  --dataset-root /data/caoqinlu/datasets/DataMind-Data \
  --output-dir /data/caoqinlu/runs/datamind_sql_smoke \
  --model DeepSeek-V4-Flash \
  --index 0-1 \
  --max-steps 10 \
  --sandbox-python /data/caoqinlu/datasets/data_agent_rl_environment_train/.venv/bin/python
```

Add `--dry-run` to validate selection and paths without calling the model.
