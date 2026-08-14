# DataCOPE ablation results

This report centralizes the completed held-out DataCOPE comparisons on
AgenticDataBench and MoSciBench. Each benchmark compares the same fixed Solving
Agent with and without the final DataCOPE-discovered skill.

## AgenticDataBench

The deterministic held-out test split contains 184 tasks. The Solver was
DSLighting ReAct with `openai/DeepSeek-V4-Flash`, thinking disabled, temperature
0, a 30-step budget and a network-disabled Docker sandbox.

| Condition | Tasks | Score | Finished | Difference |
| --- | ---: | ---: | ---: | ---: |
| ReAct control | 184 | 0.40521 | 147/184 (79.89%) | -- |
| ReAct + DataCOPE Skill | 184 | 0.33595 | 127/184 (69.02%) | -0.06926 (-6.93 pp) |

In the 184 paired tasks, the Skill condition scored higher on 28, the control
scored higher on 41 and 115 tied. This is one held-out comparison, not a
repeated estimate. The earlier provisional baseline score of `0.40059` is
superseded by the final official evaluator result `0.40521`.

Source-of-truth evaluator files:

- `data/releases/agenticdatabench_full/AgenticDataBench/testbed/results/agenticdatabench_datacope_test_baseline.json`
- `data/releases/agenticdatabench_full/AgenticDataBench/testbed/results/agenticdatabench_datacope_test_skill.json`

## MoSciBench

The held-out test contains 65 tasks across six datasets.

| Condition | Tasks | Accuracy | Correct | Difference |
| --- | ---: | ---: | ---: | ---: |
| ReAct control | 65 | 0.41538 | 27/65 | -- |
| ReAct + DataCOPE Skill | 65 | 0.32308 | 21/65 | -0.09231 (-9.23 pp) |

Dataset-level accuracy:

| Dataset | Control | DataCOPE Skill | Difference |
| --- | ---: | ---: | ---: |
| cyclone | 0.60000 | 0.30000 | -0.30000 |
| health_spa | 0.69231 | 0.69231 | 0.00000 |
| massspecgym | 0.27273 | 0.36364 | +0.09091 |
| nurse_stress | 0.36364 | 0.18182 | -0.18182 |
| pop_genetics | 0.20000 | 0.30000 | +0.10000 |
| terra | 0.30000 | 0.00000 | -0.30000 |

The machine-readable source of truth is
`experiments/datacope_moscibench/runs/test_summary.json`.

## Conclusion

The DataCOPE Skill did not improve downstream performance in either completed
held-out test. Because each benchmark has only one test comparison, these
numbers establish the observed direction but do not estimate run-to-run
variance.
