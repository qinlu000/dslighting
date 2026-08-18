# DataMind Python perception split

`split_ids.json` freezes the selected task IDs; `build_split.py` materializes
and validates the local FastPerception data split.

## Final split

| Output | Rows | Description |
|---|---:|---|
| `benchmark_100.parquet` | 100 | 44 official-test tasks + 56 moved train tasks |
| `train.parquet` | 7,154 | Python-only training tasks after source isolation |
| `official_python_test_remainder.parquet` | 276 | Other TableBench/DABench tasks, kept held out |

SQL train rows and BIRD test rows are excluded. The 56 moved tasks use 56
different files; every train task using one of those files is also excluded.
Benchmark rows contain no trajectory fields.

## Build

```bash
/data/caoqinlu/envs/datamind-sft/bin/python \
  experiments/datamind_perception/build_split.py
```

Outputs are written to `artifacts/datamind_perception/`. The builder also
writes a review manifest, an exclusion manifest, hashes, and leakage checks.
`benchmark_manifest.csv` contains references and must not be exposed to the
evaluated model.

Evaluation follows DataMind's original Python protocol: GPT-4o-mini compares
the predicted answer with the reference and returns a binary score. Training
uses the tasks and source files; new FastPerception trajectories should replace
the original DataMind trajectories.
