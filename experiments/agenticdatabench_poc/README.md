# AgenticDataBench × DSLighting PoC

This sidecar runs public AgenticDataBench tasks through a DSLighting workflow
without registering AgenticDataBench in `DSBenchmark` or modifying DSLighting's
task registry.

The boundary is intentionally small:

- AgenticDataBench owns task JSONL, domain datasets, expected output names, and scoring.
- DSLighting owns the solving workflow, sandbox, workspace, and telemetry.
- The solving agent sees only the task question and `testbed/datasets/<domain>`.
  Gold files, `eval_func`, and benchmark skill labels are not injected.

## Prerequisites

1. Clone AgenticDataBench and download its Hugging Face datasets into
   `AgenticDataBench/testbed/datasets/` as described by the upstream README.
2. Install DSLighting and the AgenticDataBench testbed requirements.
3. Configure the model API key through environment variables or the DSLighting
   repository's `.env`. Explicit environment variables take precedence.
4. Install `bubblewrap` for the default local sandbox, or install the Docker
   extra with `pip install -e '.[docker]'` for the recommended benchmark runtime.

The adapter uses the upstream `image.py` helper for plot tasks. It creates a
lightweight persistent staging view made of symlinks, so the downloaded dataset
is not modified or duplicated.

## Build the benchmark execution image

Build the exact upstream da-agent base image, then add the small set of runtime
packages needed by the network-disabled DSLighting sandbox:

```bash
docker build \
  -t da-agent:latest \
  /path/to/AgenticDataBench/testbed/da_agent/images/da-agent

docker build \
  --build-arg BASE_IMAGE=da-agent:latest \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) \
  -t dslighting-agenticdatabench:latest \
  -f experiments/agenticdatabench_poc/docker/Dockerfile .
```

The upstream Dockerfile currently contains unpinned Python packages, so a
future rebuild can resolve a different dependency set. For leaderboard runs,
publish the built base image and reference an immutable registry digest (for
example `registry.example/da-agent@sha256:...`) as `BASE_IMAGE`; record the
derived image digest alongside the run summary.

This derived benchmark image is also the recommended environment for
`mini_swe_agent`. In Docker mode DSLighting delegates command execution to
mini-swe-agent's official `DockerEnvironment`; the model client remains on the
host and only bash actions run in the benchmark container.

The DSLighting workflow and model client remain on the host. Only generated
analysis code executes in this task-scoped container. Input symlink targets are
frozen before the first action and mounted read-only; the task workspace is
mounted read-write; networking is disabled by default.

## Inspect a smoke run

Dry-run one task without calling a model:

```bash
python -m experiments.agenticdatabench_poc \
  --benchmark-root /path/to/AgenticDataBench \
  --output-dir ./runs/agenticdatabench/output/dslighting-react-smoke \
  --task agriculture_02 \
  --workflow react \
  --model openai/your-model \
  --dry-run
```

Selection is explicit to avoid accidentally spending tokens on all public
tasks. Choose one of:

- `--task TASK_ID` (repeatable)
- `--index 0-5` or `--index 0,2,4`
- `--all`

## Run

Start with a small mixed smoke set:

```bash
python -m experiments.agenticdatabench_poc \
  --benchmark-root /path/to/AgenticDataBench \
  --output-dir ./runs/agenticdatabench/output/dslighting-react-smoke \
  --task agriculture_02 \
  --task agriculture_14 \
  --workflow react \
  --model openai/your-model \
  --no-thinking \
  --sandbox-backend docker \
  --docker-image dslighting-agenticdatabench:latest \
  --concurrency 30 \
  --max-steps 30
```

The defaults are a 30-step Solving Agent budget and 30 concurrent tasks. Every
execution action may use a fresh process, while files and the container
workspace persist for the full task. `--concurrency` controls the number of
task-scoped workflows and containers that may be active at once; result
summaries remain ordered by the upstream task selection.

Provider reasoning mode is disabled by default. Comparable experiment scripts
also pass `--no-thinking` explicitly, and the resolved value is recorded as
`thinking` in the dry-run plan and run summary.

Run the same benchmark with mini-swe-agent's official Docker environment:

```bash
python -m experiments.agenticdatabench_poc \
  --benchmark-root /path/to/AgenticDataBench \
  --output-dir ./runs/agenticdatabench/output/mini-swe-agent-smoke \
  --task agriculture_02 \
  --workflow mini_swe_agent \
  --model openai/your-model \
  --sandbox-backend docker \
  --docker-image dslighting-agenticdatabench@sha256:... \
  --concurrency 1 \
  --max-steps 30
```

Resource defaults are 8192 MiB memory, 4 CPU cores, and 256 processes. Override
them with `--memory-mb`, `--cpu-cores`, and `--pids-limit` when needed.

Existing task results are skipped. Use `--retry-failed` to replace only failed
task directories or `--overwrite` to replace all selected task directories.
These flags are required before existing outputs are removed.

For a quick local debugging run without Docker or bubblewrap, pass
`--local-isolation process`. This is not recommended for comparable benchmark
results because process isolation does not enforce the default network policy.

## Perception rollout

Perception is an explicit treatment, not part of the ReAct baseline. A normal
`--workflow react` run keeps it disabled, including when Docker is used. Enable
it only for a Perception ablation arm:

```bash
python -m experiments.agenticdatabench_poc \
  --benchmark-root /path/to/AgenticDataBench \
  --output-dir ./runs/agenticdatabench/output/react-perception \
  --task agriculture_02 \
  --workflow react \
  --model openai/your-model \
  --sandbox-backend docker \
  --docker-image dslighting-agenticdatabench:latest \
  --perception
```

The Perception treatment is Docker-only, ReAct-only, and requires networking
to remain disabled. The Perception Agent and Solving Agent reuse the same
task-scoped, persistent Docker container and the same read-write workspace.

In this minimal version, the Perception Agent's no-write rule is prompt-only:
its prompt instructs it to inspect files without creating, overwriting,
deleting, or renaming anything. This is not a filesystem security boundary;
Perception code technically has the same workspace permissions as Solver code.
The shared Docker container still runs with networking disabled.

The Perception Agent uses a lightweight Action–Report protocol rather than the
Solving Agent's ReAct protocol. It returns either one `<Action>` containing a
single Python block or one plain-text `<Report>`. It never emits `<Think>` or
`<Answer>`. Action output is returned to Perception as an observation, while
the final report is wrapped in `<PerceptionResult>` for the Solving Agent.

The local sandbox, including local Bubblewrap mode, does not support Perception
in this rollout. Perception is offered only when the configured Docker backend
uses `network_mode=none`; Solver Actions remain available under the configured
sandbox.

`--max-steps` limits the Solving Agent loop. It does not include the nested
Perception turns, so report Perception usage separately when comparing
ablation arms.

### Naming candidates

- **FastPerception** — The most direct name; it clearly communicates the
  system's focus on fast, efficient perception.
- **LEAP** — **L**ightweight **E**fficient **A**gentic **P**erception. The name
  also suggests speed and forward movement, making it a strong paper-method
  candidate.

## Evaluate with AgenticDataBench

The sidecar writes the upstream-compatible layout:

```text
<output-dir>/
  agenticdatabench_tasks.jsonl
  <task-id>/
    <required output files>
    dabench/result.json
  dslighting_run_summary.json
```

Run the official evaluator from the AgenticDataBench checkout:

```bash
cd /path/to/AgenticDataBench/testbed
python3 evaluate.py \
  --output_dir /absolute/path/to/dslighting-react-smoke \
  --gold_dir gold \
  --eval_json /absolute/path/to/dslighting-react-smoke/agenticdatabench_tasks.jsonl
```

The generated JSONL contains the exact upstream records for only the tasks in
that run. It is kept outside the solving workspace, so evaluator-only fields
are available to the official scorer but are never exposed to the agent.

## Scope

This is intentionally not a full benchmark integration. It uses the internal
`DSLightingRunner.get_eval_function()` seam with a direct `execution_spec`.
Once the smoke set is stable, a full integration can add a source descriptor,
normalized registry contracts, `DSBenchmark("agenticdatabench")`, and richer
leaderboard trajectory conversion.
