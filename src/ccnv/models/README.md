# `ccnv.models`

## Overview

The `ccnv.models` package defines pure domain models, configuration containers, result dataclasses, versioned Apache Arrow schema specifications, and custom exception hierarchies for `ccnv`. It establishes the central architectural contracts across all pipeline stages, guaranteeing strict 0-based half-open interval validation, metadata serialization, and type integrity.

## Interface

- `[src/ccnv/models/config.py:ExtractConfig]`, `[src/ccnv/models/config.py:ReferenceConfig]`, `[src/ccnv/models/config.py:PredictConfig]`, `[src/ccnv/models/config.py:ReferenceQCConfig]`: Strongly-typed configuration containers for CLI orchestrators.
- `[src/ccnv/models/config.py:ReadFilters]`: Dataclass governing BAM/CRAM read filtering criteria (MAPQ, duplicates, secondary, supplementary).
- `[src/ccnv/models/config.py:parquet_path]`: Resolves the canonical `.ccnv` artifact path from a prefix.
- `[src/ccnv/models/domain.py:Contig]`: Structural contig representation holding name, length, and index.
- `[src/ccnv/models/domain.py:Cytoband]`: Cytoband chromosome interval and Giemsa staining annotation.
- `[src/ccnv/models/domain.py:SamplesheetEntry]`: Structured record for cohort samplesheet rows.
- `[src/ccnv/models/domain.py:natural_sort_key]`: Deterministic natural sorting comparator for human and organism contig naming schemes.
- `[src/ccnv/models/domain.py:select_layout]`: Resolves and validates target contigs and header ordering.
- `[src/ccnv/models/exceptions.py:CcnvError]`: Root exception for all toolkit errors, subclassed by stage-specific exceptions (`ExtractError`, `ReferenceError`, `PredictError`, `SchemaError`).
- `[src/ccnv/models/results.py:BinResult]`: Per-contig binned observed depth, expected baseline, normalized ratios, and Z-scores.
- `[src/ccnv/models/results.py:SegmentResult]`: Segment change-point interval with weighted mean ratio, mean Z-score, gain/loss call, and prenatal annotations.
- `[src/ccnv/models/results.py:PredictionOutputs]`: Comprehensive container of output paths and result data structures.
- `[src/ccnv/models/results.py:FetalFractionResult]`: Multi-resolution fetal fraction estimates (`FFY`, `FFA`, `FFX`), active fetal fraction, and sex chromosome aneuploidy results.
- `[src/ccnv/models/schemas.py:SAMPLE_SCHEMA]`: PyArrow schema definition for binned sample read count tables.
- `[src/ccnv/models/schemas.py:REFERENCE_SCHEMA]`: PyArrow schema definition for multi-branch reference model tables.
- `[src/ccnv/models/schemas.py:PrenatalModelMetadata]`: Encapsulates serialized prenatal model hyperparameter metadata, estimator coefficients, and standard error bounds.
- `[src/ccnv/models/schemas.py:validate_table]`: Validates Arrow tables against schema and metadata contracts.
- `[src/ccnv/models/schemas.py:encode_metadata]`, `[src/ccnv/models/schemas.py:decode_metadata]`: JSON-serializes and deserializes structured metadata dictionaries for Arrow table footers.

## Invariants

- `[src/ccnv/models/schemas.py:validate_table]`: All tables must strictly conform to `SAMPLE_SCHEMA` or `REFERENCE_SCHEMA` under Arrow schema version `2.0.0` with `0-based-half-open` intervals.
- `[src/ccnv/models/schemas.py:validate_metadata]`: Metadata requires non-empty `contig_lengths` mapping identifiers to positive integers; row coordinates must satisfy $0 \le \text{start} < \text{end} \le \text{length}$.
- `[src/ccnv/models/config.py:ReadFilters]`: Minimum mapping quality (`min_mapq`) must be a non-negative integer; flag booleans must be strictly typed.
- `[src/ccnv/models/domain.py:natural_sort_key]`: Human karyotype ordering places autosomes 1-22 in numerical order, followed by gonosomes X, Y, and mitochondrial M.

## Side Effects

- None

## Verification

- Run: `pytest`
