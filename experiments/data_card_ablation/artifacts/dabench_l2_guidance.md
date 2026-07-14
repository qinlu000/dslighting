### Level 2 Agent-Oriented Data Skill Card

#### Recommended Tools

Use `pandas` for CSV loading, column inspection, filtering, grouping,
missing-value handling, datetime parsing, descriptive statistics, and CSV
writing. Use `numpy` for vectorized numeric operations, masks, quantiles,
ratios, log transforms, z-scores, and finite-value checks. Use `scipy.stats` for
Pearson/Spearman correlation, p-values, t-tests, Mann-Whitney U, normality tests,
skewness, kurtosis, and z-score helpers. Use `sklearn` only for explicit
preprocessing or simple ML tasks such as train/test split, imputation, scaling,
encoding, linear/logistic regression, random forest, feature importance, and
metrics. `matplotlib` or `seaborn` may be used for quick diagnostics, but do not
rely on plots for scalar/tabular answers. Avoid heavy deep-learning, NLP,
geospatial, chemistry, raster, single-cell, or external data-fetching packages
unless explicitly required.

#### Operating Rules

First extract the concrete rule from the task wording. Prefer explicit
constraints, thresholds, formulas, split ratios, random seeds, named methods, and
requested metrics over the task title or concept label. Use concept labels only
as coarse task-family hints. If title and constraints disagree, follow the
concrete constraint.

Map task-described variables to exact input column names before computing.
Verify required columns exist. Watch for spaces, punctuation, leading/trailing
whitespace, symbols, index-like columns, missing source columns, and task-stated
fallback formulas. Inspect representative rows before trusting a column mapping.

Load tables with `pandas.read_csv`. Inspect shape, columns, dtypes, missing
values, duplicate rows, and sample rows. Check malformed strings, placeholders,
comma-separated numbers, percentages, blanks, and numeric-looking text before
conversion. Preserve row order unless sorting, grouping, or aggregation is
required.

#### Task Workflows

For scalar statistics, identify the column/subset, convert values carefully,
handle missing values as instructed, compute with `pandas`, `numpy`, or
`statistics`, and cross-check simple results.

For grouped aggregation or ranking, identify grouping keys and value columns,
check missing keys and duplicates, use `groupby`, `agg`, or `value_counts`, apply
stated tie rules, and verify the selected group against the original table.

For correlation or significance, select compared columns, drop missing pairs,
use `pandas.corr` for simple correlations, and use `scipy.stats` when p-values
or tests are required. Do not treat correlation as causation.

For distribution or normality, inspect the numeric series, remove
missing/non-finite values as instructed, use the named test if given, and check
whether sample or population statistics are expected.

For outliers, use the explicit rule if stated: IQR, z-score, percentile, or
threshold. If title and constraints disagree, follow the concrete threshold/rule.
Count and inspect affected rows before removing anything.

For feature engineering, confirm all source columns exist, apply the stated
formula exactly, keep identifiers separate, and validate new feature values on
representative rows.

For preprocessing, keep an untouched copy when comparing before/after results.
Profile missingness, duplicates, dtypes, placeholders, and malformed strings
first. Recheck shape, columns, missingness, and target statistics after cleaning.

For simple ML, only use ML when explicitly requested. Use the specified model,
split ratio, random seed, feature set, class weights, and metric. Prefer simple
CPU-safe `sklearn` models and do not upgrade to heavier models unless required.

#### Verification

Before finalizing, confirm the result uses only the provided input data, follows
task constraints and rounding rules, preserves required ordering, handles missing
values and type conversions deliberately, and can be reopened or parsed if
written to disk.
