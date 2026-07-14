# DSLighting Annotation Skill Experiment Language

This glossary defines the shared language for the MoSciBench experiment that
measures annotation-skill effects through downstream solving-agent performance.

## Language

**L1 Semantic Data Map**:
A task-independent annotation of an underlying public dataset's important data objects, variables, and structural relationships. One dataset has one map per Perception Skill Condition.
_Avoid_: task-conditioned map, solver workflow, schema dump

**Annotation Unit**:
An underlying public dataset that receives one L1 Semantic Data Map per Perception Skill Condition. Multiple benchmark tasks over the same dataset reuse the annotation.
_Avoid_: benchmark task, solver question

**Annotation Sub-agent**:
An offline agent that inspects permitted public data and produces an L1 Semantic Data Map without receiving a solver task.
_Avoid_: Solving Agent, online data profiler

**Perception Skill**:
A reusable public-data inspection capability attached only to an Annotation Sub-agent. It never appears in the Solving Agent's context.
_Avoid_: annotation arm, solver skill, workflow hint, L2 card

**Perception Skill Condition**:
An experimental condition identified by the Perception Skill attached to the Annotation Sub-agent, or by the explicit absence of an added skill in the control condition.
_Avoid_: arm, solver-context level, benchmark variant

**Downstream Experiment Condition**:
One solver-facing comparison condition: either the Main Reference Condition or one Perception Skill Condition.
_Avoid_: annotation arm, Data Card level

**No-Added-Skill Condition**:
The control Perception Skill Condition that uses the same Annotation Sub-agent setup without attaching an additional Perception Skill.
_Avoid_: no-tools condition, no-context condition, baseline skill

**Solving Agent**:
The downstream agent that receives a benchmark task and exactly one Perception Skill Condition's L1 Semantic Data Map, then produces the benchmark submission.
_Avoid_: Annotation Sub-agent, annotation generator

**Downstream Performance**:
The MoSciBench outcomes produced by a fixed Solving Agent under one Downstream Experiment Condition, including score, valid output, and missing output.
_Avoid_: annotation quality score, independent per-task evidence

**Main Reference Condition**:
The non-treatment condition that gives the Solving Agent the benchmark-provided Dataset Description instead of a generated L1 Semantic Data Map. It is not a Perception Skill Condition.
_Avoid_: main skill, baseline skill, no-added-skill condition
