# `ccnv.io`

## Overview

The `ccnv.io` package manages external format ingestion, genomic track loading, samplesheet validation, and versioned Apache Arrow/Parquet (`1.0.0`) persistence for `ccnv`. It streams single-end and paired-end templates from BAM and CRAM alignments with automatic indexing, extracts GC, mappability, and cytoband tracks from FASTA, BigWig, and tabular files, validates YAML/JSON samplesheets against `samplesheet.schema.json`, and reads/writes Zstandard-compressed `.ccnv` Parquet tables.

## Interface

- [`alignments.py:alignment_format`](./alignments.py): Maps `.bam` / `.cram` file suffixes to lowercase `"bam"` or `"cram"` (raising `ValueError` otherwise).
- [`alignments.py:ensure_index`](./alignments.py): Checks for existing `.bai` / `.crai` indices and invokes `pysam.index` when missing.
- [`alignments.py:open_alignment`](./alignments.py): Opens a `pysam.AlignmentFile` handle with htslib thread allocation and mandatory FASTA validation for CRAM inputs.
- [`alignments.py:read_alignment_header`](./alignments.py): Extracts `(contig_names, contig_lengths)` tuples from an alignment header.
- [`alignments.py:rejection_reason`](./alignments.py): Returns the deterministic QC drop counter key for filtered reads, or `None` when kept.
- [`alignments.py:stream_contig_reads`](./alignments.py) & [`stream_contig_fragments`](./alignments.py): Stream filtered 1D read starts or full paired-end/single-end fragmentomic tuples (dyad midpoints, clamped `uint16` insert lengths, endpoints, GC fractions, ``$`5'`$`` 4-mer motifs, and 1-bp histograms) across a contig.
- [`parquet.py:write_sample_parquet`](./parquet.py) & [`read_sample_parquet`](./parquet.py): Persist and validate 7-column `SAMPLE_SCHEMA` `.ccnv` Parquet files (`count`, `fragment_lengths`, `endpoint_count`, `frag_gc_sum`, plus `ccnv.*` metadata).
- [`parquet.py:read_sample_tables`](./parquet.py) & [`load_cohort`](./parquet.py): Load and validate multiple sample Parquet tables into a synchronized `Cohort` (parallelized across processes when `threads > 1`).
- [`parquet.py:is_sample_parquet`](./parquet.py) & [`load_sample_or_extract`](./parquet.py): Inspect Parquet footer metadata (`ccnv.table_kind == "sample"`) and transparently load `.ccnv`/`.parquet` samples or extract `.bam`/`.cram` alignments via a temporary workspace.
- [`parquet.py:write_reference_parquet`](./parquet.py) & [`read_reference_parquet`](./parquet.py): Persist and validate 14-column multi-branch `REFERENCE_SCHEMA` `.ccnv` Parquet files.
- [`samplesheet.py:load_samplesheet`](./samplesheet.py): Parses root-list YAML/JSON samplesheets and validates entries (`filepath`, optional `sex` in `{"F", "M"}`, optional `ff` in `(0.0, 1.0]`) against `samplesheet.schema.json`.
- [`tracks.py:read_fasta_contigs`](./tracks.py): Reads ordered `(name, length)` pairs from an indexed FASTA file via `pysam.FastaFile`.
- [`tracks.py:load_gc_track`](./tracks.py): Loads per-bin `(gc, high_n_mask)` arrays from `.fa`/`.fasta`/`.fna`, `.bw`/`.bigWig`, `.bed`/`.bed.gz`, `.bedgraph`/`.bedgraph.gz`, `.wig`/`.wig.gz`, or `.tsv`/`.txt` tracks (rescaling `[0, 100]` percentages to `[0, 1]` and setting masked/missing bins to `NaN`).
- [`tracks.py:load_mappability_track`](./tracks.py): Loads per-bin mappability scores from BigWig tracks (setting missing/non-finite intervals to `NaN` and validating finite values lie in `[0.0, 1.0]`).
- [`tracks.py:load_cytobands_track`](./tracks.py): Loads UCSC cytoband intervals and Giemsa stain labels from `.txt`/`.bed`/`.tsv` (plain or `.gz`) files.

## Invariants

- [`alignments.py:open_alignment`](./alignments.py): CRAM alignments require an existing reference FASTA file; non-positive thread counts raise `ValueError`.
- [`parquet.py:read_sample_parquet`](./parquet.py) & [`read_reference_parquet`](./parquet.py): Enforce `SCHEMA_VERSION = "1.0.0"` (`SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}`) and `0-based-half-open` coordinates.
- [`samplesheet.py:load_samplesheet`](./samplesheet.py): Rejects empty lists, duplicate resolved sample paths, non-existent sample files, and schema violations with `ReferenceError`.
- [`tracks.py:load_gc_track`](./tracks.py) & [`load_mappability_track`](./tracks.py): Finite unmasked track proportions must lie within ``$`[0.0, 1.0]`$``; bins without valid sequence or coverage receive `NaN` for downstream masking.

## Side Effects

- [`alignments.py:ensure_index`](./alignments.py) & [`tracks.py:read_fasta_contigs`](./tracks.py): Create missing `.bai`/`.crai` and `.fai` index files on disk via `pysam`.
- [`parquet.py:write_sample_parquet`](./parquet.py) & [`write_reference_parquet`](./parquet.py): Create parent directories and write Zstandard-compressed Parquet files.
- [`parquet.py:load_cohort`](./parquet.py) & [`load_sample_or_extract`](./parquet.py): Spawn `ProcessPoolExecutor` workers for parallel sample loading/rebinning (`threads > 1`) and create/remove temporary directories when extracting `.bam`/`.cram` inputs on the fly.

## Verification

- Run: `pixi run test`
