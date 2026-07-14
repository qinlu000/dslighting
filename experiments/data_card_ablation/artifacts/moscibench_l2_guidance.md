### Level 2 Scientific Data Workflow Guidance

#### Evidence and Reproducibility

Use only the provided public observations and the rules stated in the task. Do
not fetch external datasets or rely on undocumented domain facts. Before
modeling, inventory the available paths, formats, dimensions, field names,
units, coordinate systems, missing-value conventions, and identifier columns.

Prefer deterministic, CPU-safe methods. Honor any stated random seed, split,
metric, tolerance, or aggregation rule. Start with a small representative read
when a dataset is large, then confirm that the full-data path uses the same
parsing and transformations. Record assumptions in working notes and check
them against the observed values.

#### Format-Aware Inspection

Choose tools according to the data rather than forcing every object into a
table. Use `pandas` or `polars` for delimited and columnar tables, `numpy` and
`scipy` for numeric arrays and scientific calculations, and `xarray` when
labeled dimensions materially reduce axis ambiguity. Use established readers
such as `rasterio`, `tifffile`, or domain libraries only when they are already
available and match the observed format.

Inspect file signatures, schemas, shapes, dtypes, axis order, metadata, and a
few representative values before applying transformations. Treat directory
structure and paired filenames as possible relationships, but verify them from
the public evidence. Check whether zeros, sentinels, masked cells, blank
strings, or non-finite values mean absence, a measured zero, or an invalid
observation.

#### Tables, Signals, and Time Series

For tables, check duplicate rows, key uniqueness, categorical encodings,
numeric-looking strings, and units before conversion. Map task variables to
exact field names and validate joins by key cardinality rather than row count
alone.

For time-dependent data, parse timestamps deliberately, check timezone and
sampling cadence, and distinguish gaps from measured zeros. Sort only when the
operation requires it. Use chronological or group-aware splits when stated,
and avoid transformations that use information from later observations during
earlier predictions.

For sensor signals, inspect channel order, sampling rate, clipping, baseline
drift, and missing intervals. Apply filtering, windowing, resampling, or
normalization only when justified by the task and verify that these operations
do not change alignment or label boundaries.

#### Spatial, Raster, and Image Data

Confirm coordinate reference systems, affine transforms, longitude/latitude
order, pixel resolution, band order, nodata masks, and temporal indexing.
Distinguish spatial coordinates from array indices. When combining rasters or
images, verify extent, resolution, orientation, and alignment before comparing
cells.

Use masked statistics for nodata regions and inspect boundary behavior after
cropping, interpolation, reprojection, or neighborhood operations. Preserve
spatial or temporal grouping in validation when nearby observations are not
independent.

#### Spectra and Molecular Measurements

Inspect the representation of mass-to-charge values, intensities, precursor
metadata, adduct or charge fields, and collision settings. Confirm whether
spectra are centroided, normalized, sorted, or repeated. Use the task-stated
tolerance and verify whether it is absolute or relative before matching peaks.

Keep sample identifiers aligned across spectral and tabular objects. Do not
infer unavailable molecular properties from names alone. If a transformation
normalizes intensities, bins peaks, or removes low-signal observations, compare
basic counts and ranges before and after it.

#### Population and Genotype Data

Verify genotype encoding, ploidy, allele order, locus identifiers, population
labels, and the representation of missing calls. Separate individual-level,
locus-level, and population-level quantities before aggregation. Use stated
filters and estimators exactly, and check whether denominators exclude missing
observations.

Preserve population or family grouping where independence matters. Validate
frequency-like quantities against their expected numeric bounds and reconcile
simple totals on a small subset before scaling up.

#### Scientific Modeling

Use machine learning only when the task calls for it. Identify the target,
eligible predictors, grouping variables, split strategy, preprocessing, and
metric before fitting. Fit learned preprocessing on the training partition and
apply the same transformation elsewhere. Prefer a simple reproducible baseline
before adding complexity.

Watch for identifiers, repeated measurements, spatial neighbors, temporal
neighbors, or near-duplicate samples crossing partitions. Use group-aware,
spatial, or chronological validation when required by the task. Compare model
behavior with a constant, majority, or simple linear baseline appropriate to
the measured quantity.

#### Verification

Check shapes, keys, units, finite-value rates, category sets, and scientifically
meaningful bounds after each major transformation. Recompute a few cases by an
independent simple method. Stress-test empty groups, all-missing slices,
constant features, single-class partitions, duplicate identifiers, and values
at stated thresholds.

Before finalizing, confirm that every requested quantity follows the task's
exact formula, rounding, ordering, seed, and missing-value rules, and that each
claim is supported by the provided public evidence.
