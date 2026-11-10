# `ccnv.io`

## Overview

The `ccnv.io` package manages external format ingestion, track extraction, and versioned Apache Arrow/Parquet persistence for `ccnv`. It handles streaming reads from BAM and CRAM alignments with automatic indexing, loading GC and mappability tracks from FASTA, BigWig, and BED files, validating samplesheets against JSON schemas, and reading/writing zstd-compressed Parquet tables under strict coordinate and schema contracts.

## Interface

- `[src/ccnv/io/alignments.py:alignment_format]`: Inspects alignment headers to identify `"BAM"` or `"CRAM"` formats.
- `[src/ccnv/io/alignments.py:ensure_index]`: Checks for index files and calls `pysam.index` to generate `.bai` or `.crai` when missing.
- `[src/ccnv/io/alignments.py:open_alignment]`: Context manager opening `pysam.AlignmentFile` handles with reference FASTA support.
- `[src/ccnv/io/alignments.py:read_alignment_header]`: Extracts ordered contig identifiers, lengths, and raw metadata from alignment headers.
- `[src/ccnv/io/alignments.py:stream_contig_reads]`: Streams filtered 5' read start positions for a target contig.
- `[src/ccnv/io/alignments.py:rejection_reason]`: Categorizes read filter failures for quality control accounting.
- `[src/ccnv/io/parquet.py:write_sample_parquet]`: Serializes binned counts and sample QC metadata into zstd-compressed Parquet.
- `[src/ccnv/io/parquet.py:read_sample_parquet]`: Reads and validates sample tables against the `SAMPLE_SCHEMA` specification.
- `[src/ccnv/io/parquet.py:read_sample_tables]`: Loads multiple sample Parquet tables into memory.
- `[src/ccnv/io/parquet.py:write_reference_parquet]`: Serializes multi-branch reference models and scalar metadata into Parquet.
- `[src/ccnv/io/parquet.py:read_reference_parquet]`: Reads and validates reference tables against the `REFERENCE_SCHEMA` specification.
- `[src/ccnv/io/parquet.py:is_sample_parquet]`: Audits an on-disk Parquet file to verify whether it matches sample schema requirements.
- `[src/ccnv/io/parquet.py:load_cohort]`: Ingests a collection of sample Parquet paths into a structured cohort.
- `[src/ccnv/io/samplesheet.py:load_samplesheet]`: Ingests and validates YAML/JSON cohort samplesheets against `samplesheet.schema.json`.
- `[src/ccnv/io/tracks.py:load_gc_track]`: Loads per-bin GC proportions from FASTA, BigWig, or BED/TSV files.
- `[src/ccnv/io/tracks.py:load_mappability_track]`: Loads per-bin mappability scores from BigWig tracks.
- `[src/ccnv/io/tracks.py:load_cytobands_track]`: Ingests cytoband annotations for interactive genome visualization.
- `[src/ccnv/io/tracks.py:read_fasta_contigs]`: Extracts contig names and lengths from reference FASTA files via `pysam.FastaFile`.

## Invariants

- `[src/ccnv/io/alignments.py:open_alignment]`: File format must be BAM or CRAM; CRAM files strictly require a valid reference FASTA path.
- `[src/ccnv/io/parquet.py:read_sample_parquet]`: Parquet files must conform to Arrow schema version `1.0.0` with `SAMPLE_SCHEMA` layout and `0-based-half-open` intervals.
- `[src/ccnv/io/parquet.py:read_reference_parquet]`: Parquet files must conform to Arrow schema version `1.0.0` with `REFERENCE_SCHEMA` layout.
- `[src/ccnv/io/samplesheet.py:load_samplesheet]`: Samplesheet files must validate against `samplesheet.schema.json` with unique, non-empty sample IDs.
- `[src/ccnv/io/tracks.py:load_gc_track]`: GC proportions must be finite floating-point values in $[0.0, 1.0]$.
- `[src/ccnv/io/tracks.py:load_mappability_track]`: Mappability scores must be finite floating-point values in $[0.0, 1.0]$.

## Side Effects

- `[src/ccnv/io/alignments.py:ensure_index]`: Generates missing BAM (`.bai`) or CRAM (`.crai`) index files on disk via `pysam.index`.
- `[src/ccnv/io/tracks.py:read_fasta_contigs]`: Generates missing FASTA index files (`.fai`) on disk via `pysam.faidx`.
- `[src/ccnv/io/parquet.py:write_sample_parquet]`: Creates target directory hierarchies and writes zstd-compressed Parquet files to disk.
- `[src/ccnv/io/parquet.py:write_reference_parquet]`: Creates target directory hierarchies and writes zstd-compressed reference Parquet files to disk.
- `[src/ccnv/io/alignments.py:stream_contig_reads]`: Opens BAM/CRAM file descriptors and streams alignment records from disk.
- `[src/ccnv/io/tracks.py:load_mappability_track]`: Reads BigWig track files via `pyBigWig`.

## Verification

- Run: `pytest`
