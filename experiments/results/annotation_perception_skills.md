# Annotation and Perception Skills ablation

This report is the centralized result entry for the curated July 2026 DABench
and MoSciBench preliminary Annotation Skill runs. The experimental variable was
the task-independent annotation supplied to the downstream Solving Agent; the
annotation-generation skill itself was not exposed to the Solver.

The reported metric is the rate of benchmark tasks receiving score `1`. Every
task in the selected repetitions remains in the denominator, so missing numeric
scores are not silently removed.

## Selected cohorts

| Benchmark | Repetitions | Tasks per condition and repetition | Conditions |
| --- | ---: | ---: | ---: |
| DABench | 5 | 257 | 3 |
| MoSciBench | 3 | 88 | 4 |

DABench repetitions 1–5 form one protocol-homogeneous batch. For MoSciBench,
only repetitions 1–3 form the clean primary cohort. Repetition 4 changed
concurrency during its condition block; repetition 5 encountered a
deployment-wide cost limit. Those runs are excluded, and conditions from
different repetitions are not joined into a synthetic repetition.

## DABench

| Dataset-context condition | Per-repetition score=1 | Total | Rate | Difference from Original |
| --- | --- | ---: | ---: | ---: |
| Original Dataset Description | 203 / 206 / 201 / 211 / 207 | 1028/1285 | 80.0% | -- |
| Base Annotation | 198 / 205 / 209 / 198 / 203 | 1013/1285 | 78.8% | -1.2 pp |
| Semantics-Guided Annotation | 202 / 206 / 199 / 205 / 208 | 1020/1285 | 79.4% | -0.6 pp |

Neither generated-annotation condition improved the five-repetition aggregate
over the original Dataset Description on DABench.

## MoSciBench

| Dataset-context condition | Per-repetition score=1 | Total | Rate | Difference from Original |
| --- | --- | ---: | ---: | ---: |
| Original Dataset Description | 36 / 34 / 36 | 106/264 | 40.2% | -- |
| Base Annotation | 38 / 39 / 34 | 111/264 | 42.0% | +1.9 pp |
| Semantics-Guided Annotation | 34 / 38 / 42 | 114/264 | 43.2% | +3.0 pp |
| Modality-Enriched Annotation | 39 / 37 / 41 | 117/264 | 44.3% | +4.2 pp |

All three generated-annotation conditions were above the original Dataset
Description in the three-repetition MoSciBench aggregate, with the largest
observed gain for Modality-Enriched Annotation.

## Source artifacts

The canonical machine-readable index is archived outside the worktree at
`local-experiments-20260816/data-card/data-experiments/suites/annotation-skill-ablation-20260722/manifest.json`.
Original manifests, final results, metadata and mismatch logs are retained
under `local-experiments-20260816/data-card/data-experiments/archive/`.

The suite is explicitly preliminary. Its cohort rules and archive-path mapping
remain authoritative even though the numerical report is centralized here.
