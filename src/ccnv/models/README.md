# `ccnv.models`

## Overview

The `ccnv.models` package defines pure domain types, frozen configuration containers, result dataclasses, versioned Apache Arrow/Parquet (`1.0.0`) schemas, metadata serializers, and stage-specific exception classes for `ccnv`.

## Interface

- [`config.py:ExtractConfig`](./config.py), [`ReferenceConfig`](./config.py), [`ReferenceQCConfig`](./config.py), [`PredictConfig`](./config.py): Frozen configuration dataclasses for CLI and programmatic entry points, alongside `DEFAULT_BIN_SIZE` (`5000`), `DEFAULT_MIN_MAPPABILITY` (`0.5`), `AnalysisMode`, and `parquet_path`.
- [`config.py:ReadFilters`](./config.py): Alignment filtering rules (`min_mapping_quality = 1`, `exclude_duplicates = True`, `exclude_secondary = True`, `exclude_supplementary = True`, `require_proper_pair = True`).
- [`domain.py:Contig`](./domain.py), [`Cytoband`](./domain.py), [`SamplesheetEntry`](./domain.py): Immutable genomic layout records (`Contig` stores `name`, `length`, and `role` in `{"autosomal", "sex_baseline", "sex_signal"}`).
- [`domain.py:select_layout`](./domain.py), [`is_canonical_contig`](./domain.py), [`normalize_chr`](./domain.py), [`natural_sort_key`](./domain.py): Canonical chromosome selection, prefix normalization, and natural contig sorting.
- [`exceptions.py:CcnvError`](./exceptions.py): Base toolkit error subclassed by `ExtractError`, `ReferenceError`, `ReferenceQCError`, `PredictError`, `CovariateError`, and `SchemaError`.
- [`results.py:BinResult`](./results.py) & [`GenomeProfile`](./results.py): Per-branch normalized bin arrays (`contigs`, `starts`, `ends`, `ratios`, `zscores`, `weights`, `reference_sizes`, `masked`, `branch`) and multi-branch genome profile containers.
- [`results.py:SegmentResult`](./results.py): Segmented interval record holding weighted `ratio`, autocorrelation-adjusted Stouffer `zscore`, `weight`, `bin_count`, `call` (`"gain"` / `"loss"` / `None`), prenatal ratio/discordance fields, and fragmentomic origin fields (`frag_short_ratio`, `frag_llr_size`, `origin_call`).
- [`results.py:SCAResult`](./results.py), [`FetalFractionResult`](./results.py), [`PredictionOutputs`](./results.py), [`ReferenceQCOutputs`](./results.py): Prenatal sex chromosome aneuploidy, multi-modal fetal fraction (`ffy`, `ffa`, `ffx`, `ff_size`, `active_ff`, `active_ff_source`), and pipeline deliverable containers.
- [`schemas.py:SAMPLE_SCHEMA`](./schemas.py) & [`REFERENCE_SCHEMA`](./schemas.py): PyArrow table schemas under `SCHEMA_VERSION = "1.0.0"` (`SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}`), plus branch constants (`autosomal`, `female_sex`, `male_sex`, `prenatal_autosomal`, `prenatal_ffy`).
- [`schemas.py:FragmentomicsMetadata`](./schemas.py), [`PrenatalModelMetadata`](./schemas.py), [`ReferenceMetadata`](./schemas.py): Serializable metadata dataclasses for sample fragmentomic landmarks/histograms, prenatal model weights/densities/standard errors, and reference provenance.
- [`schemas.py:encode_metadata`](./schemas.py), [`decode_metadata`](./schemas.py), [`validate_table`](./schemas.py), [`validate_metadata`](./schemas.py), [`validate_bin_intervals`](./schemas.py), [`validate_reference_columns`](./schemas.py): Arrow footer metadata codecs and structural schema/coordinate validators.

## Invariants

- [`schemas.py:validate_table`](./schemas.py): Enforces exact `SAMPLE_SCHEMA` (7 columns) or `REFERENCE_SCHEMA` (14 columns) field types, `SCHEMA_VERSION = "1.0.0"`, and `0-based-half-open` coordinate bounds ``$`0 \le \text{start} \lt \text{end} \le \text{length}`$``.
- [`config.py:ReadFilters`](./config.py): `min_mapping_quality` is an integer threshold and flag filters are strict booleans.
- [`domain.py:natural_sort_key`](./domain.py): Tokenizes stripped chromosome names so numeric chunks sort before case-folded text chunks (`1..22` before `M`, `X`, `Y`).

## Side Effects

- None

## Verification

- Run: `pixi run test`
