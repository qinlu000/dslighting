# Artifact inventory

## `manifests/`

- `training_mix_2000.jsonl`: canonical cross-source task manifest.
- `dare_bench_600.jsonl`: upstream DARE records needed by its runner.
- `data_agent_rl_700_ids.json`: selected Data Agent task IDs.
- `datamind_sql_700_ids.json`: selected DataMind SQL task IDs.

The manifests store references to the locally downloaded benchmark data; they
do not duplicate database archives or ground-truth files.

The current pipeline trains on all 2,000 rows in the canonical manifest and
uses DABench and AgenticDataBench for OOD evaluation. The old `splits/`
directory is retained only as a historical artifact and is not consumed by
the code.

## `annotations/`

- `taxonomy_annotations.jsonl`: two-pass DeepSeek decisions, agreement fields,
  optional adjudication, official DARE overrides, evidence spans, confidence,
  and final primary/secondary labels.

## `audits/`

- `selection_summary.json`: deterministic selection and evaluator checks.
- `similarity_summary.json`: duplicate/near-duplicate screening summary.
- `similarity_candidates.csv`: candidates inspected by the similarity audit.
- `taxonomy_audit.json`: reproducible integrity and agreement summary.
- `taxonomy_review_queue.csv`: low-confidence and unresolved manual-review set.

## `figures/`

- `taxonomy_task_donut.{png,pdf,svg}`: current semantic task composition.
- `taxonomy_primary_counts.csv`: exact values plotted in the figure.

## `cold_start/`

- `sft_v1_qwen3_24k/`: the frozen 207-row Solver-only cold-start baseline used
  by the first 4B/8B SFT models and the current 4B GRPO run.
- `teacher_candidates_400.jsonl`: the frozen teacher pool used by that
  Solver-only baseline.
- `teacher_candidates_900.jsonl`: the complete frozen teacher task pool used
  to build the joint dataset.
- `perception_retry_round_*.jsonl`: the frozen targeted retry assignments used
  to increase valid Perception coverage without repeating already covered tasks.
- `perception_sft_final_medium_t09_qwen3_24k/clean/`: the canonical 276 Solver
  and 235 Perception records plus the Perception acceptance audit.
- `perception_sft_final_medium_t09_qwen3_24k/llamafactory/`: the directly
  trainable 511-row ShareGPT dataset and dataset registry. The SHA-256 of
  `train.json` is
  `409a16d978f3ef685f52649d5d75fa172e67780d4799efd37971e880778203d8`.
- `perception_sft_final_medium_t09_qwen3_24k/export_summary.json`: frozen input
  runs, role/source counts, rejection reasons, and token/integrity checks.

The large per-round export directories are not retained. The frozen teacher
pool, retry manifests, final acceptance audit, and final export summary preserve
the selections and provenance needed to interpret the canonical dataset.
