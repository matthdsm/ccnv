# `ccnv.io`

## Overview

The `ccnv.io` package manages external format ingestion, genomic track loading, samplesheet validation, and versioned Apache Arrow/Parquet (`2.0.0`) persistence for `ccnv`. It streams single-end and paired-end templates from BAM and CRAM alignments via the compiled Rust `ccnv._core` scanner, extracts GC, mappability, and cytoband tracks from FASTA, BigWig, and tabular files, validates YAML/JSON samplesheets against `samplesheet.schema.json`, and reads/writes Zstandard-compressed `.ccnv` Parquet tables.

## Interface

- [`alignments.py:alignment_format`](./alignments.py): Maps `.bam` / `.cram` file suffixes to lowercase `"bam"` or `"cram"` (raising `ValueError` otherwise).
- [`alignments.py:read_alignment_header`](./alignments.py): Extracts `(contig_names, contig_lengths)` tuples from an alignment header via `ccnv._core`.
- [`alignments.py:scan_alignment_bins`](./alignments.py): Streams filtered 1D read starts or full paired-end/single-end fragmentomic channels (dyad midpoints, clamped insert lengths, 24-band distributions, endpoints, GC fractions, $`5'`$ 4-mer motifs, and 1-bp histograms) across contigs.
- [`parquet.py:write_sample_parquet`](./parquet.py) & [`read_sample_parquet`](./parquet.py): Persist and validate 7-column coordinate-free `SAMPLE_SCHEMA` `.ccnv` Parquet files (`count`, `paired_count`, `frag_len_sum`, `short_frag_count`, `endpoint_count`, `frag_gc_sum`, `frag_bands`, plus `ccnv.*` metadata).
- [`parquet.py:read_sample_tables`](./parquet.py), [`extract_sample_meta_and_counts`](./parquet.py), & [`load_cohort`](./parquet.py): Load and validate sample Parquet metadata (`ccnv.artifact_kind == "sample"`) and assemble synchronized `Cohort` buffers (parallelized across worker threads when `threads > 1`).
- [`parquet.py:write_reference_parquet`](./parquet.py) & [`read_reference_parquet`](./parquet.py): Persist and validate 17-column multi-branch `REFERENCE_SCHEMA` `.ccnv` Parquet files with per-branch row groups.
- [`samplesheet.py:load_samplesheet`](./samplesheet.py): Parses root-list YAML/JSON samplesheets and validates entries (`filepath`, optional `sex` in `{"F", "M"}`, optional `ff` in `(0.0, 1.0]`) against `samplesheet.schema.json`.
- [`tracks.py:read_fasta_contigs`](./tracks.py): Reads ordered `(name, length)` pairs from an indexed FASTA file via `noodles-fasta` (`ccnv._core`).
- [`tracks.py:load_gc_track`](./tracks.py): Loads per-bin GC proportions and high-N masks from FASTA (`ccnv._core`), BigWig (`bigtools`), or tabular tracks.
- [`tracks.py:load_mappability_track`](./tracks.py): Loads per-bin mappability scores from BigWig tracks via `bigtools` in `ccnv._core` (validating finite values lie in `[0.0, 1.0]`).
- [`tracks.py:load_cytobands_track`](./tracks.py): Loads UCSC cytoband intervals and Giemsa stain labels from `.txt`/`.bed`/`.tsv` (plain or `.gz`) files.

## Invariants

- [`alignments.py:read_alignment_header`](./alignments.py): CRAM alignments require an existing reference FASTA file; non-positive thread counts raise `ValueError`.
- [`parquet.py:read_sample_parquet`](./parquet.py) & [`read_reference_parquet`](./parquet.py): Enforce `SCHEMA_VERSION = "2.0.0"` (`SUPPORTED_SCHEMA_VERSIONS = {"2.0.0"}`) and `0-based-half-open` coordinates.
- [`samplesheet.py:load_samplesheet`](./samplesheet.py): Rejects empty lists, duplicate resolved sample paths, non-existent sample files, and schema violations with `ReferenceError`.
- [`tracks.py:load_gc_track`](./tracks.py) & [`load_mappability_track`](./tracks.py): Finite unmasked track proportions must lie within $`[0.0, 1.0]`$; bins without valid sequence or coverage receive `NaN` for downstream masking.

## Side Effects

- [`parquet.py:write_sample_parquet`](./parquet.py) & [`write_reference_parquet`](./parquet.py): Create parent directories and write Zstandard-compressed Parquet files.
- [`parquet.py:load_cohort`](./parquet.py): Spawns worker threads during parallel sample loading.

## Verification

- Run: `pixi run test`
