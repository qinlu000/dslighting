# Earth-observation and environmental data

Use these checks for event tracks, station or grid time series, remote-sensing
arrays, rasters, imagery, and mixed metadata-plus-scene collections.

## Observation units and axes

- Distinguish event-level, track-point, station-time, grid-cell-time,
  scene-time, pixel, and tile observations.
- Identify timestamp meaning, cadence, timezone or calendar when stated, and
  whether values are instantaneous, averaged, accumulated, or interval-based.
- Identify latitude/longitude order, longitude range, coordinate units, grid
  spacing, coordinate reference system, spatial extent, and axis direction only
  when supported by metadata or values.
- Distinguish fractions from percentages and verify the scale of masks, quality
  fields, coverage values, and interpolation indicators from observed values or
  metadata.
- For wide grid tables, determine whether columns encode named locations,
  coordinates, bands, or something else; do not assume a column position is a
  longitude or latitude.

## Images, rasters, arrays, and containers

- Inspect image or array shape, dtype, channel or band count, dimension names,
  scale and offset attributes, nodata values, and physical units when available.
- For HDF5-like containers, inspect groups, dataset names, shapes, dtypes, and
  attributes. Do not infer axis order from shape alone.
- Distinguish display imagery from calibrated physical measurements. Pixel
  values, color channels, and scientific bands are not interchangeable.
- Treat coordinate-encoded filenames, world files, projection sidecars, and
  metadata tables as evidence that must agree; a filename alone does not prove
  geolocation.

## Relationships to capture

- Link each scene, raster, array, or time-series partition to its metadata using
  explicit IDs, paths, timestamps, coordinates, or documented naming rules.
- Verify that metadata-referenced files exist, and compare the key coverage of
  tables, grids, images, and containers before claiming complete alignment.
- State whether an event has repeated observations, whether a grid is shared
  across times, and whether metadata-to-file relations are one-to-one or
  one-to-many.
- Record dateline wrapping, duplicated boundary cells, missing tiles, irregular
  time steps, or inconsistent path references when observed.

Describe these semantics without recommending interpolation, reprojection,
feature extraction, forecasting, or visualization.
