---
name: perceive-scientific-modalities
description: "Add format- and modality-aware inspection to task-independent scientific dataset annotation. Use alongside a general dataset-perception workflow when public artifacts contain earth-observation data, physiological sensing, mass spectra and molecular records, or population-genetics bundles."
---

# Perceive Scientific Modalities

Add non-obvious scientific format and modality checks to a general semantic
data-mapping workflow. Preserve the caller's evidence boundary, output schema,
and prohibition on solver guidance.

## Route by observed evidence

Identify modalities from file signatures, container metadata, schemas, field
families, shapes, and representative values. Do not route from a benchmark task,
answer target, dataset family label, or hidden description.

Read every reference that matches the observed public artifacts:

- Earth-observation tracks, grids, imagery, rasters, or environmental time
  series: read [earth-observation.md](references/earth-observation.md).
- Physiological, wearable, environmental-sensor, session, or survey streams:
  read [physiological-sensing.md](references/physiological-sensing.md).
- Peak lists, spectra, precursor metadata, molecular formulas, SMILES, or
  InChIKeys: read [mass-spectrometry.md](references/mass-spectrometry.md).
- PLINK bundles, genotype matrices, variant tables, pedigrees, or cohort
  metadata: read [population-genetics.md](references/population-genetics.md).

Load multiple references when a dataset combines modalities. If none match,
perform only the general dataset-perception workflow; do not force a scientific
interpretation.

## Apply the modality pass

1. Identify the modality's observation unit, native axes, format-defined
   metadata, units, sentinel values, and logical sidecar bundle.
2. Find the object that anchors alignment across modalities, such as a subject,
   session, timestamp, location, event, spectrum, sample, or variant identifier.
3. Check whether ordering is semantically significant and whether aligned files
   share the same IDs, coordinates, timestamps, sample order, or variant order.
4. Distinguish raw measurements from metadata, annotations, identifiers, and
   alternate representations of the same underlying object.
5. Record ambiguities instead of filling them with customary domain defaults.
   A common convention is evidence only when the observed format establishes
   that convention.

Use only already available, read-only inspection tools. Do not install a domain
library, contact an external database, or execute a solver workflow. Express the
additional findings through the caller's existing semantic-map fields.
