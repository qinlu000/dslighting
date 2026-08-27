# Data-agent training mixture (2,000 tasks)

This directory is the paper-facing, self-contained record of the curated
training mixture built from DARE-Bench, Data Agent RL Environment, and DataMind
SQL. Every selected task has evaluator-only ground truth and a programmatic
scorer; no LLM judge is required for training reward.

## Directory layout

```text
paper/training_mix/
├── README.md
├── code/
│   ├── build_manifest.py                 # deterministic 2k selection
│   ├── classify_taxonomy_deepseek.py     # constrained DeepSeek annotation
│   ├── summarize_taxonomy.py             # integrity/agreement audit
│   ├── plot_task_categories.py           # paper figure from semantic labels
│   ├── run_local_perception_teacher.sh   # parameterized teacher rollout
│   ├── sft.py                            # shared SFT record rules
│   ├── export_perception_cold_start_sft.py # reward-gated SFT export
│   ├── assemble_perception_augmented_sft.py # assemble the frozen 511 rows
│   └── trajectory_audit.py               # shared protocol/data safety gates
├── config/
│   ├── taxonomy_v1.yaml                  # definitions and precedence rules
│   └── sft_qwen3_4b_perception_joint_zero2_24k.yaml
├── artifacts/
│   ├── manifests/                        # selected task records and source IDs
│   ├── annotations/                      # per-task taxonomy decisions
│   ├── audits/                           # selection, similarity, taxonomy QA
│   ├── figures/                          # current paper-ready figures/tables
│   └── cold_start/                       # frozen 207- and 511-row SFT data
└── tests/
    ├── test_taxonomy.py
    ├── test_export_perception_cold_start_sft.py
    └── test_trajectory_audit.py
```

## Frozen mixture

| Source | Tasks | Selection notes |
|---|---:|---|
| DARE-Bench | 600 | 275 classification, 275 regression, 50 forecasting |
| Data Agent RL | 700 | programmatic answer modes only; runtime-heavy share capped |
| DataMind SQL | 700 | executable SQLite tasks with CSV ground truth |
| **Total** | **2,000** | unique task IDs and normalized prompts |

The canonical task manifest is
`artifacts/manifests/training_mix_2000.jsonl`. The complete taxonomy result is
`artifacts/annotations/taxonomy_annotations.jsonl`.

## Training and evaluation split

All 2,000 curated tasks are used for training. DABench and AgenticDataBench
are held out as OOD benchmarks; no in-domain task is reserved for checkpoint
selection.

## Perception cold start

The canonical joint Solver/Perception SFT dataset is:

```text
artifacts/cold_start/perception_sft_final_medium_t09_qwen3_24k/
```

It contains 511 accepted trajectories: 276 Solver and 235 Perception records.
The final export combines a frozen 900-task teacher pool with targeted retries
on tasks that had not yet produced a valid Perception trajectory. The complete
input run list, acceptance counts, rejection reasons, maximum token length, and
train-file SHA-256 are frozen in `export_summary.json`.

Teacher rollout requests disable provider-internal thinking while retaining
the visible `<Think>` agent protocol. Export is evaluator-gated rather than
teacher-judged: DARE requires score at least 0.5, while Data Agent RL and
DataMind SQL require exact reward 1. Solver records must also pass that task
gate. Perception records may come from an unsuccessful Solver episode, but
must be complete, protocol-valid, read-only, free of execution-error
observations, unique by trajectory and task, and no longer than 24,576 Qwen3
tokens.

`run_local_perception_teacher.sh` is a parameterized raw-rollout launcher, not
a second canonical dataset definition. The deterministic post-processing path
is implemented by `export_perception_cold_start_sft.py` and
`assemble_perception_augmented_sft.py`; the frozen teacher and retry manifests
record the selections from the completed rollout rounds. Shared trajectory
gates live in `trajectory_audit.py`.

The local 4B training configuration is
`config/sft_qwen3_4b_perception_joint_zero2_24k.yaml`. The HPC 4B and 8B
configurations are under `experiments/training_mix/hpc/`. Both trained models
use the same 511-row file and have Qwen provider thinking disabled.

## Taxonomy design

`config/taxonomy_v1.yaml` defines one mutually exclusive primary analytical
objective plus overlapping secondary objectives, operation tags, reasoning
complexity, domain-knowledge dependence, and requested output forms. Only the
primary objective is used for a part-to-whole plot.

The separation avoids mixing semantic tasks with implementation (`SQL`,
Python), evaluator formats (`numeric`, `exact_short`), or reasoning attributes
(`multi-hop`, domain-specific). The codebook is grounded in DataMind's
fine-grained analytical categories, DARE-Bench's official predictive modeling
types, and CRISP-DM's data-understanding/preparation/modeling/evaluation stages.

## Reproduction

All commands run from the repository root.

### 1. Annotate or resume taxonomy classification

The default is prompt-only classification with two independently ordered
DeepSeek passes. Primary/secondary disagreement triggers a third adjudication.
DARE primary labels are overridden by official benchmark metadata while the
model's pre-override decision is retained for auditing.

```bash
.venv/bin/python -m paper.training_mix.code.classify_taxonomy_deepseek \
  --model DeepSeek-V4-Flash \
  --concurrency 100 \
  --passes 2
```

`API_KEY`, `API_BASE`, and optional `API_HOST_OVERRIDES` are read from `.env`.
Provider thinking is disabled. Output is checkpointed by task ID, so rerunning
resumes only missing or failed tasks. Add `--overwrite` only when intentionally
replacing all annotations.

Optional DataMind SQL evidence:

```bash
--include-reference-sql \
--datamind-parquet /data/caoqinlu/datasets/DataMind-Data/rl/train.parquet
```

### 2. Re-run the annotation audit

```bash
.venv/bin/python -m paper.training_mix.code.summarize_taxonomy
```

This verifies one-to-one task IDs, prompt hashes, structured labels, and exact
evidence spans, then writes:

- `artifacts/audits/taxonomy_audit.json`;
- `artifacts/audits/taxonomy_review_queue.csv`.

### 3. Rebuild the semantic task figure

```bash
.venv/bin/python -m paper.training_mix.code.plot_task_categories
```

The inner ring shows analytical lifecycle stages; the outer ring shows unified
primary task objectives across all three sources. Dataset source and scorer
formats are not used as task categories.

### 4. Run tests

```bash
.venv/bin/pytest -q paper/training_mix/tests
```

## Current annotation QA

- 2,000/2,000 annotations pass structured, hash, and evidence validation;
- two-pass primary-label agreement: 92.35%;
- 316 tasks required automatic adjudication;
- 75 tasks are retained in the manual-review queue;
- 6 tasks remain explicitly `unresolved` rather than being forced into a class.
