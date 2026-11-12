# `ccnv.models`

## Overview

The `ccnv.models` package defines pure domain types, frozen configuration containers, result dataclasses, zero-copy in-memory sample and branch array builders, versioned Apache Arrow/Parquet (`2.0.0`) schemas, metadata models and serializers, validation routines, and stage-specific exception classes for `ccnv`.

## Interface

- [`buffers.py:SampleBuffers`](./buffers.py), [`BranchFitArrays`](./buffers.py), & [`assemble_reference_table`](./buffers.py): Zero-copy NumPy-backed sample buffer containers (`SampleBuffers.from_table`, `SampleBuffers.coerce`, `SampleBuffers.to_table`) and reference branch table builders.
- [`config.py:ExtractConfig`](./config.py), [`ReferenceConfig`](./config.py), [`ReferenceQCConfig`](./config.py), [`PredictConfig`](./config.py): Frozen configuration dataclasses for CLI and programmatic entry points, alongside `DEFAULT_BIN_SIZE` (`5000`), `DEFAULT_MIN_MAPPABILITY` (`0.5`), `AnalysisMode`, `SexBranch`, and `parquet_path`.
- [`config.py:ReadFilters`](./config.py): Alignment filtering rules (`min_mapping_quality = 1`, `exclude_duplicates = True`, `exclude_secondary = True`, `exclude_supplementary = True`, `exclude_unmapped = True`, `require_proper_pair = True`).
- [`domain.py:ContigLookup`](./domain.py), [`Cytoband`](./domain.py), [`SamplesheetEntry`](./domain.py): Immutable genomic layout records and $`O(1)`$ normalized chromosome resolvers.
- [`domain.py:select_layout`](./domain.py), [`is_canonical_contig`](./domain.py), [`is_autosome_name`](./domain.py), [`normalize_chr`](./domain.py), [`natural_sort_key`](./domain.py): Canonical chromosome selection, prefix normalization, and natural contig sorting.
- [`exceptions.py:CcnvError`](./exceptions.py): Base toolkit error subclassed by `ExtractError`, `ReferenceError`, `ReferenceQCError`, `PredictError`, `CovariateError`, `SchemaError`, `SchemaVersionError`, `AlignmentError`, `BinCompatibilityError`, and `InvalidInputError`.
- [`metadata.py:SampleMetadata`](./metadata.py), [`ReferenceMetadata`](./metadata.py), [`FragmentomicsMetadata`](./metadata.py), [`PrenatalModelMetadata`](./metadata.py): Serializable dataclasses for sample provenance, reference hyperparameters, fragmentomics landmarks, and prenatal models (`female_chrx_ratio_mean`, `female_chrx_ratio_std`, `alpha_1`, `band_fisher_weights`, `mu_fetal_w`, `mu_maternal_w`), alongside `METADATA_PREFIX`, `FRAGMENTOMICS_METADATA_KEY`, and `PRENATAL_METADATA_KEY`.
- [`metadata.py:encode_metadata`](./metadata.py) & [`decode_metadata`](./metadata.py): Arrow schema metadata codecs for versioned JSON encoding and decoding.
- [`results.py:BinResult`](./results.py), [`ScanAlignmentResult`](./results.py), & [`ScanQcCounts`](./results.py): Per-branch normalized bin arrays (`contigs`, `starts`, `ends`, `ratios`, `zscores`, `weights`, `reference_sizes`, `masked`, `branch`) and typed Rust alignment scanner return payloads.
- [`results.py:SegmentResult`](./results.py) & [`CallType`](./results.py): Segmented interval record holding weighted `ratio`, dyadic BLUE `zscore`, `weight`, `bin_count`, `call` (`"gain"` / `"loss"` / `None`), discrete `copy_number`, `log_bf`, `q_value`, prenatal ratio/discordance fields, and fragmentomic origin fields (`frag_short_ratio`, `frag_llr_size`, `origin_call`, `mosaic_fraction`, `mosaic_ci`).
- [`results.py:SCAResult`](./results.py), [`SCACall`](./results.py), [`FetalSex`](./results.py), [`ActiveFFSource`](./results.py), [`BranchQCStatus`](./results.py), [`PrenatalLoDMetrics`](./results.py), [`FetalFractionResult`](./results.py), [`PredictionOutputs`](./results.py), & [`ReferenceQCOutputs`](./results.py): Prenatal sex chromosome aneuploidy, limit-of-detection metrics, multi-modal fetal fraction (`ffy`, `ffa`, `ffx`, `ff_size`, `active_ff`, `active_ff_se`, `active_ff_source`, `effective_ff`, `lod_metrics`), branch QC status records, and pipeline deliverable containers.
- [`schemas.py:SAMPLE_SCHEMA`](./schemas.py), [`REFERENCE_SCHEMA`](./schemas.py), [`ArtifactKind`](./schemas.py), [`build_reference_schema`](./schemas.py), [`reconstruct_sample_grid`](./schemas.py), [`canonical_dictionary_array`](./schemas.py), [`decode_dictionary_contigs`](./schemas.py), [`unique_dictionary_contigs`](./schemas.py), & [`fixed_size_list_to_numpy`](./schemas.py): PyArrow table schemas under `SCHEMA_VERSION = "2.0.0"` (`SUPPORTED_SCHEMA_VERSIONS = {"2.0.0"}`, `COORDINATE_SYSTEM = "0-based-half-open"`), dimension constants (`NUM_FRAG_BANDS`, `NUM_GC_QUANTILES`, `NUM_PROTOTYPES_PER_BIN`), Arrow dictionary/list zero-copy helpers, and branch constants (`KNOWN_REFERENCE_BRANCHES`, `BRANCH_AUTOSOMAL`, `BRANCH_FEMALE_SEX`, `BRANCH_MALE_SEX`, `BRANCH_PRENATAL_AUTOSOMAL`, `BRANCH_PRENATAL_FFY`).
- [`validation.py:validate_sample_table`](./validation.py), [`validate_reference_table`](./validation.py), [`validate_schema`](./validation.py), & [`validate_metadata`](./validation.py): Arrow table, schema, and metadata structural validators.

## Invariants

- [`validation.py:validate_sample_table`](./validation.py) & [`validate_reference_table`](./validation.py): Enforce exact `SAMPLE_SCHEMA` (7 coordinate-free per-bin columns) or `REFERENCE_SCHEMA` (17 columns) field types, `SCHEMA_VERSION = "2.0.0"`, and `0-based-half-open` coordinate bounds $`0 \le \text{start} \lt \text{end} \le \text{length}`$.
- [`config.py:ReadFilters`](./config.py): `min_mapping_quality` is an integer threshold and flag filters are strict booleans.
- [`domain.py:natural_sort_key`](./domain.py): Tokenizes stripped chromosome names so numeric chunks sort before case-folded text chunks (`1..22` before `M`, `X`, `Y`).

## Side Effects

- None

## Verification

- Run: `pixi run test`
