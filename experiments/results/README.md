# Experiment results

This directory contains only compact, aggregate experiment results suitable for
version control. Task outputs, evaluator payloads, submissions, prompts,
message histories, workspaces, telemetry and execution trajectories are not
included. Reports record the local source-artifact paths for provenance only.

[`summary.json`](summary.json) provides the consolidated machine-readable
result index.

## Current AgenticDataBench benchmark

- [AgenticDataBench three-run ablation](agenticdatabench_three_run_ablation.md)
- [Superseded AgenticDataBench trials](agenticdatabench_legacy_trials.md)

AgenticDataBench is a benchmark. The current runner and FastPerception
treatment live under `experiments/agenticdatabench/`.

## Retired method studies

- [DataCOPE ablations](datacope_ablation.md)
- [Data Card level ablations](data_card_level_ablation.md)
- [Annotation and Perception Skills ablation](annotation_perception_skills.md)

DataCOPE and Data Card are retired methods, not benchmark integrations. Their
aggregate results remain for scientific provenance, while code, workspaces and
raw trajectories are kept outside the active worktree.
