# `ccnv.models`

## Overview

The `ccnv.models` package defines pure domain types, frozen configuration containers, result dataclasses, versioned Apache Arrow/Parquet (`2.0.0`) schemas, metadata models and serializers, validation routines, and stage-specific exception classes for `ccnv`.

## Interface

- [`config.py:ExtractConfig`](./config.py), [`ReferenceConfig`](./config.py), [`ReferenceQCConfig`](./config.py), [`PredictConfig`](./config.py): Frozen configuration dataclasses for CLI and programmatic entry points, alongside `DEFAULT_BIN_SIZE` (`5000`), `DEFAULT_MIN_MAPPABILITY` (`0.5`), `AnalysisMode`, and `parquet_path`.
- [`config.py:ReadFilters`](./config.py): Alignment filtering rules (`min_mapping_quality = 1`, `exclude_duplicates = True`, `exclude_secondary = True`, `exclude_supplementary = True`, `require_proper_pair = True`).
- [`domain.py:ContigLookup`](./domain.py), [`Cytoband`](./domain.py), [`SamplesheetEntry`](./domain.py): Immutable genomic layout records and $`O(1)`$ normalized chromosome resolvers.
- [`domain.py:select_layout`](./domain.py), [`is_canonical_contig`](./domain.py), [`is_autosome_name`](./domain.py), [`normalize_chr`](./domain.py), [`natural_sort_key`](./domain.py): Canonical chromosome selection, prefix normalization, and natural contig sorting.
- [`exceptions.py:CcnvError`](./exceptions.py): Base toolkit error subclassed by `ExtractError`, `ReferenceError`, `ReferenceQCError`, `PredictError`, `CovariateError`, and `SchemaError`.
- [`metadata.py:SampleMetadata`](./metadata.py), [`ReferenceMetadata`](./metadata.py), [`FragmentomicsMetadata`](./metadata.py), [`PrenatalModelMetadata`](./metadata.py): Serializable dataclasses for sample provenance, reference hyperparameters, fragmentomics landmarks, and prenatal models.
- [`metadata.py:encode_metadata`](./metadata.py) & [`decode_metadata`](./metadata.py): Arrow schema metadata codecs for versioned JSON encoding and decoding.
- [`results.py:BinResult`](./results.py): Per-branch normalized bin arrays (`contigs`, `starts`, `ends`, `ratios`, `zscores`, `weights`, `reference_sizes`, `masked`, `branch`).
- [`results.py:SegmentResult`](./results.py): Segmented interval record holding weighted `ratio`, dyadic BLUE `zscore`, `weight`, `bin_count`, `call` (`"gain"` / `"loss"` / `None`), discrete `copy_number`, `log_bf`, `q_value`, prenatal ratio/discordance fields, and fragmentomic origin fields (`frag_short_ratio`, `frag_llr_size`, `origin_call`, `mosaic_fraction`).
- [`results.py:SCAResult`](./results.py), [`FetalFractionResult`](./results.py), [`PredictionOutputs`](./results.py), [`ReferenceQCOutputs`](./results.py): Prenatal sex chromosome aneuploidy, multi-modal fetal fraction (`ffy`, `ffa`, `ffx`, `ff_size`, `active_ff`, `active_ff_se`, `active_ff_source`), and pipeline deliverable containers.
- [`schemas.py:SAMPLE_SCHEMA`](./schemas.py), [`REFERENCE_SCHEMA`](./schemas.py), [`SampleBuffers`](./schemas.py), [`BranchFitArrays`](./schemas.py), & [`assemble_reference_table`](./schemas.py): PyArrow table schemas under `SCHEMA_VERSION = "2.0.0"` (`SUPPORTED_SCHEMA_VERSIONS = {"2.0.0"}`), zero-copy in-memory sample/reference builders, and branch constants (`autosomal`, `female_sex`, `male_sex`, `prenatal_autosomal`, `prenatal_ffy`).
- [`validation.py:validate_table`](./validation.py), [`validate_metadata`](./validation.py), [`validate_bin_intervals`](./validation.py), [`validate_reference_columns`](./validation.py): Arrow table and schema structural validators.

## Invariants

- [`validation.py:validate_table`](./validation.py): Enforces exact `SAMPLE_SCHEMA` (7 coordinate-free per-bin columns) or `REFERENCE_SCHEMA` (17 columns) field types, `SCHEMA_VERSION = "2.0.0"`, and `0-based-half-open` coordinate bounds $`0 \le \text{start} \lt \text{end} \le \text{length}`$.
- [`config.py:ReadFilters`](./config.py): `min_mapping_quality` is an integer threshold and flag filters are strict booleans.
- [`domain.py:natural_sort_key`](./domain.py): Tokenizes stripped chromosome names so numeric chunks sort before case-folded text chunks (`1..22` before `M`, `X`, `Y`).

## Side Effects

- None

## Verification

- Run: `pixi run test`
