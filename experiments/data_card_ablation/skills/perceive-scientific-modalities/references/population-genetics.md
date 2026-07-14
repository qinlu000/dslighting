# Population genetics and PLINK-style bundles

Use these checks for genotype matrices, PLINK BED/BIM/FAM bundles, pedigree or
sample tables, variant metadata, and multiple cohort prefixes.

## Logical bundle

- Group files with the same prefix when they jointly define one genotype
  dataset. Treat the binary genotype matrix, variant table, and sample or
  pedigree table as distinct components of that logical object.
- Inspect binary format markers and orientation metadata when available. Verify
  matrix sample and variant counts against the companion sample and variant
  tables instead of trusting suffixes or filenames.
- For standard PLINK-style files, verify rather than assume the usual roles:
  BIM-like rows describe chromosome, variant ID, genetic position, base-pair
  position, and two allele codes; FAM-like rows describe family ID, individual
  ID, parental IDs, sex code, and phenotype code.
- Identify extra or updated sample tables as alternate metadata versions, not
  automatically as additional individuals. Detect header-like first rows and
  original, updated, or backup variants before counting records or choosing a
  canonical component.

## Alignment and encoding

- Preserve sample order and variant order as alignment axes for the binary
  matrix. Check counts and matching prefixes when available without decoding
  private or forbidden artifacts.
- Distinguish family ID from individual ID and record how supplementary cohort
  tables join to the bundle.
- Treat zeroes, negative values, and numeric phenotype or sex codes as format or
  dataset-specific encodings. Do not assign their meanings without evidence.
- Do not label the two allele columns as reference, alternate, effect, ancestral,
  or derived alleles unless metadata explicitly establishes that role.
- Record uncertainty about genome build, chromosome-code convention, strand,
  genotype orientation, and missing-call encoding when those facts are absent.

## Multiple cohorts

- Keep distinct cohort or population bundles separate while representing their
  shared schema and any explicit cross-cohort metadata.
- Do not infer ancestry, relatedness, case-control status, or population labels
  from filenames, genotypes, or conventional codes alone.

Describe genotype and cohort semantics without recommending PCA, association
testing, imputation, quality-control thresholds, or ancestry analysis.
