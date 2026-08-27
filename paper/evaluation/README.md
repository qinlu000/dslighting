# OOD evaluation protocol

The external OOD evaluations are DABench and AgenticDataBench. Neither uses an
LLM judge in this pipeline. DABench uses task-specific deterministic graders;
AgenticDataBench uses its published evaluator functions and gold artifacts.

Across the 2,000 training prompts and all 503 public OOD prompts, normalized
exact overlap is zero. A character 3--5 gram TF-IDF screen finds no cross-split
pair at cosine similarity 0.70 or above; the maximum is 0.4393. The reproducible
summary is stored in `ood_overlap_audit.json`.

## DABench

- local source: `/data/caoqinlu/datasets/DABench`;
- DSLighting task data: `/data/caoqinlu/datasets/DABench/data`;
- 257 of 257 tasks are materialized;
- all 257 private answers score 1.0 against their matching graders.

The DSLighting `DSBenchmark` interface is the runtime entry point. A real
DeepSeek-V4-Flash → ReAct → Bubblewrap → deterministic-grader smoke on
`dabench-0-mean-fare-paid` scored 1.0.

## AgenticDataBench

- checkout: `/data/caoqinlu/datasets/AgenticDataBench`;
- pinned public checkout commit:
  `61bb0d6be3439797d2c75a6ede198b0b296cc226`;
- 246 public tasks across 15 domains;
- all declared public input paths and gold paths resolve;
- the upstream 98-task private test set is withheld and is not downloaded.

Use the isolated Python 3.10 interpreter at
`/data/caoqinlu/datasets/AgenticDataBench/.venv/bin/python` through Bubblewrap.
The standard full public run is:

```bash
AGENTICDATABENCH_PYTHON=/data/caoqinlu/datasets/AgenticDataBench/.venv/bin/python \
  .venv/bin/python -m experiments.agenticdatabench \
  --benchmark-root /data/caoqinlu/datasets/AgenticDataBench \
  --output-dir /absolute/run/output \
  --all --model MODEL_NAME --max-steps 10
```

The official evaluator gives only 223 of the 246 published gold outputs a
perfect gold-on-gold score. The 23 exact task IDs and scores are recorded in
`agenticdatabench_gold_selfscore_audit.json`. For a self-consistent diagnostic
subset, add:

```bash
--exclude-task-list-json \
  paper/evaluation/agenticdatabench_gold_selfscore_exclude_23.json
```

Always report the unmodified 246-task public result separately; never present
the filtered 223-task diagnostic subset as the upstream public benchmark. A
real DeepSeek-V4-Flash → ReAct → Bubblewrap → official-evaluator smoke on
`financial_83` scored 1.0.

## HPC pipeline validation

The frozen 2,000-task mixture passed compute-node preflight, a 32-task/two-GPU
training smoke, and a 96-task/four-GPU pilot with three optimizer updates. The
pilot completed 384 of 384 rollouts, preserved exactly four samples for each
of 96 tasks, and wrote a complete `global_step_3` FSDP checkpoint. Every audit
reward event used a programmatic scorer, and all 2,617 model requests disabled
provider-internal thinking.

The merged-model HPC smoke then ran both OOD paths against a real local vLLM
server. DABench scored 1.0 on `dabench-0-mean-fare-paid`. AgenticDataBench's
official evaluator completed on `strategy_3` and assigned 0.0 to the model's
incorrect artifact; that zero is a model-quality result, not an evaluator
failure.

The reproducible job IDs and metrics are recorded in
`pipeline_audit_2026-08-26.json`. The full 257+246 task HPC command is in
`experiments/training_mix/HPC_RUNBOOK.md`; it uses the same vLLM, Bubblewrap,
and official programmatic graders as the smoke.
