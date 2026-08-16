# DABench and MoSciBench Data Card level ablation

This report centralizes the completed June 2026 Data Card level runs. Each
benchmark was evaluated with AIDE and ReAct under four historical run labels:
`l0`, `l1`, `l2` and `l3`. The historical `l0` label corresponds to the current
`main` policy: the original benchmark Dataset Description with no additional
L1 map or L2 guidance.

The model was `DeepSeek-V4-Flash`. AIDE used a maximum of 3 iterations and
ReAct used 10. Each benchmark/workflow/condition cell has one run, so these
tables do not estimate run-to-run variance.

## Scores

`Actual score` is the metadata field `score.actual_average`: total score divided
by every benchmark task, including unscored tasks as zero. This is the value to
compare across conditions.

| Benchmark | Workflow | Tasks | main (`l0`) | L1 | L2 | L3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DABench | AIDE | 257 | 0.87549 | 0.88327 | 0.88327 | 0.89494 |
| DABench | ReAct | 257 | 0.80156 | 0.81323 | 0.78988 | 0.79377 |
| MoSciBench | AIDE | 88 | 0.60227 | 0.65909 | 0.63636 | 0.65909 |
| MoSciBench | ReAct | 88 | 0.32955 | 0.37500 | 0.36364 | 0.36364 |

Difference from `main`:

| Benchmark | Workflow | L1 - main | L2 - main | L3 - main |
| --- | --- | ---: | ---: | ---: |
| DABench | AIDE | +0.00778 | +0.00778 | +0.01946 |
| DABench | ReAct | +0.01167 | -0.01167 | -0.00778 |
| MoSciBench | AIDE | +0.05682 | +0.03409 | +0.05682 |
| MoSciBench | ReAct | +0.04545 | +0.03409 | +0.03409 |

## Scored-task coverage

The corresponding number of tasks with numeric scores was:

| Benchmark | Workflow | main (`l0`) | L1 | L2 | L3 |
| --- | --- | ---: | ---: | ---: | ---: |
| DABench | AIDE | 257/257 | 256/257 | 256/257 | 257/257 |
| DABench | ReAct | 237/257 | 241/257 | 236/257 | 235/257 |
| MoSciBench | AIDE | 75/88 | 77/88 | 67/88 | 76/88 |
| MoSciBench | ReAct | 40/88 | 45/88 | 41/88 | 40/88 |

The conditional `score.average` values that exclude unscored tasks are not used
for treatment comparisons because coverage differs substantially by condition.

## Interpretation

- AIDE improved over its main condition at every Data Card level on both
  benchmarks; its largest observed gains were DABench L3 (+1.95 pp) and
  MoSciBench L1/L3 (+5.68 pp).
- ReAct improved with L1 on both benchmarks, while DABench L2 and L3 were below
  main in this single run.
- These are historical one-run cells with uneven scored-task coverage. They are
  useful directional evidence, not a repeated estimate of a stable effect.

## Source artifacts

The source metadata, task-level result CSVs, submissions and paired-run
manifests are archived outside the worktree under
`local-experiments-20260816/legacy-benchmark-runs/runs/`:

- `dabench_aide_paired_20260628_122304` plus the L3 continuation
  `dabench_aide_paired_20260629_132443`;
- `dabench_react_paired_20260629_090725` plus the L3 continuation
  `dabench_react_paired_20260629_132443`;
- `moscibench_aide_paired_20260628_151941` plus the L3 continuation
  `moscibench_aide_paired_20260629_132217`;
- `moscibench_react_paired_20260629_090903` plus the L3 continuation
  `moscibench_react_paired_20260629_132233`.
