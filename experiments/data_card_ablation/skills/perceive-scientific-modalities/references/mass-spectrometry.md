# Mass spectrometry and molecular records

Use these checks for spectra, peak lists, precursor metadata, molecular
identifiers, formulas, structures, and compound-linked tables.

## Spectrum structure

- Determine whether one row represents a spectrum, compound, acquisition, or
  another unit. Identify stable spectrum and molecule identifiers separately.
- Treat m/z and intensity peak lists as paired ragged arrays. Check that their
  lengths and ordering correspond within representative records. Check for
  empty spectra, duplicate or unsorted m/z values, and non-finite numbers.
- Determine whether intensities are raw, scaled, normalized, or relative only
  when values or metadata support the claim. Zero-to-one values do not alone
  prove a particular normalization rule.

## Precursor and molecule semantics

- Distinguish fragment m/z values, precursor m/z, neutral or parent mass,
  precursor formula, molecular formula, adduct, and charge. Do not treat these
  as interchangeable numeric or textual fields.
- Distinguish a structure string, molecular formula, connectivity identifier,
  and stereochemistry-bearing identifier. Record whether stereochemistry or
  charge information is absent rather than reconstructing it.
- Check whether multiple spectra share a molecule identifier and whether one
  spectrum has multiple candidate representations.

## Encoding and uncertainty

- Record delimiters, numeric precision, empty-list and missing-value encodings,
  and any mismatch between paired peak arrays.
- Do not infer instrument type, collision conditions, ionization mode, compound
  identity, or chemical correctness unless public metadata establishes it.
  Confirm that a claimed metadata field is physically present rather than
  assuming it belongs to a customary spectrum schema.

Describe spectral and molecular semantics without recommending library search,
similarity scoring, peak filtering, structure prediction, or identification.
