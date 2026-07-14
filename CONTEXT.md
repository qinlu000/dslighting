# DSLighting Annotation Skill Experiment Language

This glossary defines the shared language for the MoSciBench experiment that
measures annotation-skill effects through downstream solving-agent performance.

## Language

**L1 Semantic Data Map**:
A task-independent annotation of an underlying public dataset's important data objects, variables, and structural relationships. One dataset has one map per Annotation Arm.
_Avoid_: task-conditioned map, solver workflow, schema dump

**Annotation Unit**:
An underlying public dataset that receives one L1 Semantic Data Map per Annotation Arm. Multiple benchmark tasks over the same dataset reuse the annotation.
_Avoid_: benchmark task, solver question

**Annotation Sub-agent**:
An offline agent that inspects permitted public data and produces an L1 Semantic Data Map without receiving a solver task.
_Avoid_: Solving Agent, online data profiler

**Annotation Data Perception Skill**:
A reusable public-data inspection capability attached only to an Annotation Sub-agent. It never appears in the Solving Agent's context.
_Avoid_: solver skill, workflow hint, L2 card

**Annotation Arm**:
An experimental condition whose Annotation Sub-agent setup differs from other arms only by its attached Annotation Data Perception Skills.
_Avoid_: solver-context level, benchmark variant

**No-Added-Skill Arm**:
The Annotation Arm that uses the same Annotation Sub-agent setup without attaching an additional Annotation Data Perception Skill.
_Avoid_: no-tools arm, no-context arm

**Solving Agent**:
The downstream agent that receives a benchmark task and exactly one arm's L1 Semantic Data Map, then produces the benchmark submission.
_Avoid_: Annotation Sub-agent, annotation generator

**Downstream Performance**:
The MoSciBench outcomes produced by a fixed Solving Agent under one Annotation Arm, including score, valid output, and missing output.
_Avoid_: annotation quality score, independent per-task evidence

**Reference Condition**:
An optional non-treatment condition using the benchmark-provided Dataset Description for comparison. It is not an Annotation Arm.
_Avoid_: baseline skill arm, no-added-skill arm
