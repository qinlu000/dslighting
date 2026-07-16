---
name: perceive-dataset-semantics
description: "Inspect an explicitly declared set of public dataset artifacts and construct a grounded, task-independent semantic data map. Use for offline Annotation Sub-agents that must identify logical objects, variables, units, axes, keys, granularity, encodings, and relationships without seeing solver tasks or recommending analysis workflows."
---

# Perceive Dataset Semantics

Construct a compact semantic map of what a dataset contains and how its parts
relate. Derive every claim from permitted public artifacts, and separate
observations from inferences.

## Preserve the annotation boundary

- Work only from the supplied annotation packet and its declared public input
  artifacts. If the caller declares one input file, do not scan its parent or
  sibling directories for additional context.
- Treat benchmark questions, target answers, private data, gold outputs, grader
  code, output contracts, submission templates, and hidden reference
  descriptions as forbidden evidence. Do not open or use them even if they are
  reachable in the workspace.
- Do not modify source data, access the network, install packages, or expand the
  permitted evidence boundary to compensate for a missing reader.
- Describe what the data objects, variables, and relationships mean. Do not
  provide loaders, code, plots, derived answers, modeling choices, preprocessing
  recipes, or recommended solver steps.
- Keep the result independent of any downstream task. Do not select facts
  because they appear useful for a particular benchmark question.
- Follow the caller's annotation schema exactly. Do not introduce a competing
  schema from this skill.
- Use only caller-declared logical relative paths in the result. Never expose
  an absolute host path, clean-release path, task directory, or representative
  task ID.

## Inspect the dataset

1. **Fix the evidence boundary.** Identify the dataset ID, public root, declared
   input manifest, excluded artifacts, and output schema before inspecting
   contents. If the packet exposes task-specific or forbidden material, ignore
   it and record the boundary problem without using that material.

2. **Build a physical inventory.** Inspect only declared paths, formats,
   directory groupings, and repeated naming patterns. Distinguish real inputs
   from output templates using the caller's manifest. Treat an extension as a
   format hint, not proof of content. Do not put file sizes or row counts into
   the semantic map.

3. **Group logical data objects.** Combine sidecars, partitions, per-subject or
   per-period files, and container members when they jointly represent one
   conceptual object. Keep separate objects separate when they have different
   observation units or roles. A file is not automatically a data object.

4. **Inspect bounded, representative content.** Use read-only tools
   to inspect headers, schemas, shapes, dtypes, metadata, container attributes,
   and small samples. Sample across partitions rather than assuming the first
   file represents all files. Use counts, ranges, and value patterns only to
   understand encodings and semantics, not to perform downstream analysis. Do
   not add counts, frequencies, extrema, ranges, correlations, distribution
   conclusions, outliers, or model results to the semantic map. Small observed
   values may support an encoding interpretation, but do not report their
   frequency or turn them into downstream findings.

5. **Infer semantics from evidence.** Prefer evidence in this order:

   - embedded metadata, data dictionaries, and format-defined fields;
   - consistent structure and values observed across files;
   - field names and units supported by observed values;
   - directory and filename conventions;
   - domain inference.

   Never promote a lower-confidence clue over contradictory higher-confidence
   evidence. Leave a unit null and record uncertainty rather than guessing.
   Treat identifiers, category codes, row numbers, and sequence indices as
   dimensionless unless the public evidence establishes a physical unit.

6. **Map structure explicitly.** Determine the observation unit and granularity
   of each object; identifiers and candidate keys; temporal, spatial, channel,
   or array axes; partitioning and ordering; join or alignment relationships;
   missing-value encodings; and one-to-one, one-to-many, or repeated-measure
   structure where grounded. Verify that referenced files actually exist and
   compare key coverage across related objects; a populated path or ID does not
   prove that its counterpart is present.

7. **Cover semantic variables.** Include identifiers, coordinates, axes,
   measurements, categorical encodings, and other fields needed to understand
   each important object. For an ordinary bounded table, cover every source
   column. Summarize only genuinely repetitive high-dimensional variable
   families when enumerating every member would obscure the structure. Preserve
   source column names exactly, and do not omit a field because it resembles a
   downstream target.

8. **Calibrate uncertainty.** Keep an internal distinction among observed,
   format-defined, inferred, and unknown facts. State useful unresolved
   ambiguities in the annotation's uncertainty field. Do not turn uncertainty
   into speculative prose.

## Produce the semantic map

- Give each logical object a stable, non-empty name and a semantic meaning more
  informative than its file format.
- Describe observation granularity, candidate identifiers, ordering, grouping,
  repeated-measure structure, category encodings, and missing conventions only
  when grounded in declared evidence.
- Set a variable's unit to `null` when the evidence does not establish one.
- Put useful ambiguity in `uncertainties`; keep process notes short and
  evidence-bound.
- Return only the caller-requested artifact. Do not include an explanation,
  work trace, Markdown fence, or alternative schema.

## Audit the result

Before returning the annotation, verify that:

- every referenced path belongs to the permitted public manifest;
- every variable belongs to a named data object and, when applicable, a
  concrete source file;
- logical bundles and cross-file alignment are represented rather than reduced
  to disconnected file summaries;
- units, axes, keys, missing encodings, and granularity are stated only when
  supported;
- no hidden description, private artifact, output template, task, answer, or
  grader fact influenced the annotation;
- no sentence tells a solving agent what to compute, fit, predict, visualize,
  save, or submit; and
- no exact dataset result statistic appears in the map;
- the final artifact is concise, schema-valid, and task-independent; and
- the final artifact contains no fields outside the caller's schema.
