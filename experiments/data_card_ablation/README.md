# Data Card Ablation

This entry runs one benchmark task set under any selected subset of four
solver-context policies. The bundled profiles support DABench and MoSciBench;
one invocation selects exactly one source with `--benchmark`. There is no L0
arm: the `main` baseline already contains the benchmark's dataset description,
so a second inventory-style baseline would duplicate it.

| Policy | Dataset-description slot | Data Profile | Extra guidance |
| --- | --- | --- | --- |
| `main` | Original benchmark text | Main behavior, unchanged | None |
| `l1` | Replaced by the selected L1 semantic map | Same as `main` | None |
| `l2` | Original benchmark text | Same as `main` | Benchmark-specific L2 Markdown |
| `l3` | Replaced by the selected L1 semantic map | Same as `main` | Same L2 Markdown |

Data Perception remains an internal producer used by the normal Data Profile.
Neither is an experiment arm, and both follow the same main path in all four
policies. Output Contract is owned by the selected benchmark profile and stays
identical across all arms; task-context policy never changes it.

Within one invocation, every policy uses the same deterministic result basename
for a given task while writing into its own policy log directory. This keeps
the solver-visible Data Profile and I/O text fixed across arms.

## Benchmark boundary

Benchmark discovery is catalog-driven. A source must use the MLE task contract
and MLE engine to participate; the ablation layer does not branch on task-ID
prefixes. The shared contract supplies registry construction, canonical task
layout, and dataset identity to the experiment. The benchmark profile supplies
only experiment-level defaults and reviewed assets, while the normal benchmark
registry remains authoritative for task descriptions, public data, sample
templates, and evaluation references.
DABench defaults to `aide`; MoSciBench defaults to its current `react` profile.
An explicit `--workflow` override applies uniformly to every selected arm.

The data root must contain the selected source's prepared competitions. Both a
root that directly contains task directories and a release root whose
`competitions/` child contains them are accepted. Task selection is checked
against the catalog-built registry, including the normal public, sample, and
private grading readiness checks, before any arm starts.

Every benchmark uses its resolved canonical `competition.public_dir` directly.
The ablation does not introduce another data directory or execution path:
workspace construction, data linking, and sandbox behavior remain the same as
main. Only dataset-description composition inside `TaskContextBuilder` varies
by policy.

The two bundled sources use different section headings in their task text:
DABench uses `## Data Description`, while MoSciBench uses
`## Dataset Description`. The context builder replaces exactly one recognized
section and fails closed if it is missing or ambiguous.

## L1 semantic maps

L1 is supplied as a directory of researcher-reviewed JSON artifacts. Artifact
identity comes from the resolved benchmark layout rather than from CLI string
conventions. DABench resolves task-level identities. The MoSciBench manifest
maps `task_metadata.source_family` to `mosci-<family>`, so tasks over the same
public dataset reuse one reviewed family artifact. Use a dry run to inspect the
resolved artifact IDs before launching a full experiment.

When `l1` or `l3` is selected, preflight checks every selected task before any
policy starts. Checks include the core L1 schema, resolved dataset identity,
visible-public-data coverage, and rejection of sample templates. Runtime
repeats the same boundary before composing a solver context.

A minimal DABench artifact is:

```json
{
  "schema_version": "l1_semantic_map_v1",
  "dataset_id": "dabench-example",
  "task_id": "dabench-example",
  "data_objects": [
    {
      "name": "records",
      "files": ["input.csv"],
      "kind": "table",
      "meaning": "Public observations used by the task."
    }
  ],
  "variables": [],
  "structure": [],
  "uncertainties": [],
  "annotation_notes": []
}
```

Extra fields and duplicate JSON keys are rejected. Every visible public input
must be covered, while sample templates and paths outside the public root are
forbidden. Solver-visible L1 prose also rejects private, hidden, grader, answer,
submission, Output Contract, and main-owned task-artifact I/O content.

L1/L2 files are researcher-reviewed experiment inputs. The lexical checks are
defense in depth for obvious leakage, not a proof about arbitrary natural
language. Generate them only from public observations, review them, and use the
manifest digest to pin the reviewed version. In particular, do not derive a
MoSciBench semantic map from `prepare.py`, task metadata, answer data, or
evaluation code.

## L2 guidance

L2 is intentionally source-specific. DABench defaults to
[`artifacts/dabench_l2_guidance.md`](artifacts/dabench_l2_guidance.md), while
MoSciBench defaults to
[`artifacts/moscibench_l2_guidance.md`](artifacts/moscibench_l2_guidance.md).
The latter covers format-aware workflows across scientific domains rather than
reusing DABench's table-oriented assumptions. Use `--l2-guidance-path` to pin a
different reviewed artifact for the selected source.

These files are experiment assets; production code does not import them. They
are body content and may use H3-or-deeper headings because the runtime owns the
single `## Level 2 Guidance` wrapper. Private/grader facts, submission or
Output Contract content, and main-owned task-artifact I/O instructions are
rejected in both headings and prose.

## MoSciBench prepared-release compatibility

The current standard MoSciBench prepared release is supported directly. The
catalog resolves each task's real `competition.public_dir`, and all policies
use the same established main workspace and sandbox path.

For `l1` and `l3`, the L1 artifact must match the selected release's resolved
dataset identity and cover every canonical public input at its actual relative
path. The selected task description must also accept deterministic L1 section
replacement. Reuse an artifact only when its release and path layout still
match; otherwise regenerate and review it before running. The runner does not
translate artifacts produced for another release or layout.

## Run

Configure the model credentials understood by `ConfigBuilder`, then run
DABench with an explicit source:

```bash
python experiments/data_card_ablation/run_ablation.py \
  --benchmark dabench \
  --data-root /path/to/dabench \
  --l1-artifact-dir /path/to/dabench-l1-artifacts \
  --model openai/gpt-5 \
  --concurrency 8
```

Run MoSciBench against its standard prepared release with release/layout-matched
L1 artifacts and the default L2 guidance:

```bash
python experiments/data_card_ablation/run_ablation.py \
  --benchmark moscibench \
  --data-root /path/to/moscibench \
  --l1-artifact-dir /path/to/moscibench-l1-artifacts \
  --model openai/gpt-5 \
  --concurrency 8
```

All four policies run by default. `--policies` accepts a comma-separated subset
of `main,l1,l2,l3`; duplicates and every other value, including `l0`, are
rejected. Artifact requirements follow the selected subset, so a baseline-only
plan needs neither L1 nor L2 input:

```bash
python experiments/data_card_ablation/run_ablation.py \
  --benchmark moscibench \
  --data-root /path/to/moscibench \
  --policies main \
  --model openai/gpt-5 \
  --dry-run
```

Choose tasks inline or from a file:

```bash
python experiments/data_card_ablation/run_ablation.py \
  --benchmark dabench \
  --data-root /path/to/dabench \
  --l1-artifact-dir /path/to/dabench-l1-artifacts \
  --tasks dabench-24-mean-individuals-dataset,dabench-10-total-traded-quantity \
  --limit 1 \
  --model openai/gpt-5 \
  --dry-run
```

`--tasks-file` accepts one task ID per line, with blank lines and `#` comments
ignored. Without either task option, all contracts in the selected source
registry are selected; `--limit` is applied afterward.

The runner validates every artifact family required by the selected policies
and builds all selected `ConfigBuilder` configs before executing. Each selected
policy is then run through its own `DSBenchmark` instance. A failure stops
subsequent policies instead of falling back to main context when it is an
artifact, context-composition, runtime-infrastructure, or audit-invariant
failure. Solver and evaluation outcomes are recorded and do not stop later
arms.

Every non-dry invocation writes
`experiments/data_card_ablation/runs/<run-id>/manifest.json`. The manifest
records the benchmark source, task selection, runtime settings, artifact
digests, and each selected policy's status, benchmark result paths, per-task
context audit, workflow outcome, and evaluation outcome. After each policy,
the runner verifies the data-report digest, final I/O digest, and result
basename against the first completed arm; a mismatch fails the experiment.
Task failures are explicit in the manifest and yield the overall status
`completed_with_task_failures`. A dry run prints the validated JSON plan to
stdout without creating a run directory or starting `DSBenchmark.run`.
