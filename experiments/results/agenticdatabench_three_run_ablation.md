# AgenticDataBench three-run ablation results

This document records the final merged results of the 2026-08-04
AgenticDataBench experiment. It compares the plain ReAct control with two
separate treatments: a frozen task-independent Datacard and FastPerception.
Datacard and FastPerception were never enabled together.

## Protocol

| Setting | Value |
| --- | --- |
| Benchmark | AgenticDataBench public full set, 246 tasks per run |
| Repetitions | 3 per condition; 9 runs and 2,214 task-runs total |
| Solver model | `openai/DeepSeek-V4-Flash`, HKUST-GZ API |
| Sampling | temperature `0`; provider thinking disabled |
| Solver workflow | DSLighting ReAct |
| Solver budget | 50 steps |
| Context | 320,000 max history characters; keep 50 recent turns; summarize from turn 51 |
| Original concurrency | 100 task-scoped workflows |
| API retries / task timeout | 30 / 3,600 seconds |
| Sandbox | `dslighting-agenticdatabench:latest`, Docker network disabled |
| Output contract gate | Disabled |
| Datacard treatment | 39 frozen public-root annotations, reused by tasks in the same domain |
| FastPerception treatment | Read-only-by-prompt Action–Report agent; at most 4 nested perception steps |

All three Datacard repetitions used the same canonical set of cards. Its
canonical content hash was
`9bce7d8f30644f404bb6a2d589ad5e2717a7cb239e899cb6ea879d05de894119`.
The solver-facing rendering did not expose `Level 1`, `L1`, schema-version or
condition metadata.

The Docker image was referenced by the mutable `:latest` tag, and run metadata
did not pin the repository Git commit. These are reproducibility limitations;
future comparable runs should record an immutable image digest and commit.

## Official scores

The score and finished rate below come from the AgenticDataBench official
evaluator after infrastructure repairs were merged. `Finished` means that all
benchmark-required files existed; it is not the same as receiving full credit.

| Round | Condition | Score | Finished |
| --- | --- | ---: | ---: |
| R1 | ReAct | 0.41654 | 231/246 (93.90%) |
| R1 | ReAct + Datacard | 0.41875 | 225/246 (91.46%) |
| R1 | FastPerception | 0.43262 | 236/246 (95.93%) |
| R2 | ReAct | 0.45587 | 239/246 (97.15%) |
| R2 | ReAct + Datacard | 0.45129 | 232/246 (94.31%) |
| R2 | FastPerception | 0.43131 | 242/246 (98.37%) |
| R3 | ReAct | 0.42844 | 234/246 (95.12%) |
| R3 | ReAct + Datacard | 0.45692 | 232/246 (94.31%) |
| R3 | FastPerception | 0.44435 | 235/246 (95.53%) |

### Three-run aggregate

The reported standard deviation is the sample standard deviation across the
three run-level scores.

| Condition | Mean score | Sample SD | Mean finished | Difference from ReAct |
| --- | ---: | ---: | ---: | ---: |
| ReAct | 0.43361 | 0.02017 | 95.39% | -- |
| ReAct + Datacard | 0.44232 | 0.02061 | 93.36% | +0.00871 (+0.87 pp) |
| FastPerception | 0.43609 | 0.00718 | 96.61% | +0.00248 (+0.25 pp) |

At 246 tasks per run, the mean score differences correspond to approximately
2.14 full-score-task equivalents per round for Datacard and 0.61 for
FastPerception. Across all 738 paired task-runs, Datacard scored higher than
ReAct on 156, lower on 141 and tied on 441. FastPerception scored higher on
153, lower on 129 and tied on 456.

The treatment effects were not consistent across repetitions:

| Round | Datacard - ReAct | FastPerception - ReAct |
| --- | ---: | ---: |
| R1 | +0.00222 | +0.01608 |
| R2 | -0.00457 | -0.02456 |
| R3 | +0.02848 | +0.01591 |

With only three repetitions, a run-level t interval is necessarily wide. The
95% intervals for the mean differences were `[-0.03466, 0.05208]` for
Datacard and `[-0.05569, 0.06064]` for FastPerception. Both include zero.
Accordingly, the supported conclusion is:

> Datacard averaged +0.87 percentage points across three runs, but variance was
> high and more independent clean repetitions are needed. FastPerception
> averaged +0.25 percentage points and is likewise inconclusive.

## Step-budget termination

A task-run is counted as budget-terminated only when it produced 50 Solving
Agent replies without a valid final `<Answer>`. Four additional task-runs
returned a valid answer on step 50 and are not counted below.

| Condition | R1 | R2 | R3 | Total |
| --- | ---: | ---: | ---: | ---: |
| ReAct | 12 | 9 | 9 | 30/738 |
| ReAct + Datacard | 8 | 7 | 6 | 21/738 |
| FastPerception | 9 | 4 | 8 | 21/738 |
| All conditions | 29 | 20 | 23 | 72/2,214 (3.25%) |

Of the 72 budget-terminated task-runs, 25 had already produced all required
files and remained eligible for official scoring; 47 were unfinished. The 72
instances covered 37 distinct benchmark task IDs.

## Infrastructure repair and merge

The original runs encountered provider-side `No deployments available` and
related transport failures. Repairs selected only missing, malformed or
transport-contaminated task results. A valid retry replaced the matching
`task_id` in the original 246-task output; it was never appended as an extra
sample. Legitimate solver failures, including missing required output files,
were accepted rather than retried based on score.

| Round | ReAct replacements | Datacard replacements | FastPerception replacements |
| --- | ---: | ---: | ---: |
| R1 | 2 | 4 | 5 |
| R2 | 12 | 10 | 15 |
| R3 | 63 | 51 | 58 |

R1 repair used concurrency 10; R2 and R3 repair used concurrency 20. After the
merge, every one of the nine runs had 246 unique task IDs and zero missing
results, malformed results or remaining LLM transport errors. The official
evaluator was then rerun once over each merged 246-task directory.

## Data integrity and source of truth

All nine final task manifests contained the same 246 unique task IDs and the
same ordered-ID hash:
`87c3aaf27a80561c60888fce467cc67787502474563900e67c58287bb0a18f7f`.
Every evaluation JSON contained 246 numeric scores in `[0, 1]`, and its
reported average exactly matched arithmetic over those entries.

Use the official evaluation JSON files under
`runs/agenticdatabench/evaluations/hkust-v4flash-parallel-c100-*.json` as the
source of truth. In particular, some R1 `dslighting_run_summary.json` files
retain stale resume-oriented `skipped` aggregates even though those tasks have
valid final results. Their official evaluation score and finished rate are not
affected.

The final repair audit is recorded at
`runs/agenticdatabench/retries/parallel-c100-until-clean-20260803/status.json`.
It completed on 2026-08-04 at 14:33:55 Asia/Shanghai with zero residual issues.
Retry output copies were subsequently deleted after merge; accepted-ID lists,
logs, workspace messages and telemetry were retained locally outside Git.
Completed execution `sandbox/` trees were removed after final scoring because
their required task outputs had already been copied into the final output
directories.
