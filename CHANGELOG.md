# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Replaced `pysam` and `samtools` dependencies with `polars-bio` for reading BAM/CRAM and FASTA indexes.
- Dropped automatic index generation for BAM/CRAM and FASTA; execution fails fast with an actionable error if a required index (such as `.fai` for CRAM) is absent.
- Updated `ccnv convert` to stream alignments with Polars-Bio and forward thread budgets to DataFusion multithreading.
- Unified default bin size to 5,000 bp across `convert`, `reference`, and models.
- `ccnv reference` accepts FASTA inputs as a positional argument alongside sample Parquet inputs, removing the `--fasta` option.
- `ccnv reference` rebins sample counts to `--bin-size` when cohort samples differ in native bin size, requiring matching contig lengths.
- `ccnv predict` rebins sample counts in memory to match the reference bin coordinates when bin sizes differ, without modifying the reference file.
- Convert and FASTA reference now share one contig-layout selector (`select_layout`)
  so canonical filtering is no longer duplicated, and `GenomeProfile` is the
  in-memory layout type. Prediction I/O lives in `ccnv.report`, with result
  types in `ccnv.results`. Bin coordinates are read through `schemas.bin_coordinates`.
- CLI tests compare Rich output after stripping ANSI so CI color does not
  split flags such as ``--loglevel``.

### Fixed

- Resolved `UnboundLocalError` for `fasta` variable in `ccnv reference` CLI command.

### Added

- `rebin_counts` and `rebin_sample_table` for 1D mass-conserving count interpolation across bin grids.
- `ccnv convert`: stream BAM/CRAM into versioned sample Parquet, auto-index
  missing `.bai`/`.crai`, bin canonical chromosomes (or `--all-contigs`), and
  split `--threads` across contig workers and htslib decode.
- `ccnv reference`: build autosomal and optional sex-contig models from control
  Parquet, or a flat uniform-depth model from FASTA when there is no cohort.
- `ccnv predict`: PCA-normalize, segment with weighted Binseg, call gains and
  losses, and write 0-based half-open BED, optional SVG, and matching Parquet.
- Versioned Parquet contracts (`1.0.0`), header-authoritative contig layouts,
  and organism-agnostic naming (no fixed 24-chromosome genome).
- Sex-contig branches inferred from a Y-signal contig (`--sex-cutoff` optional);
  no per-sample sex labels on the CLI.
- CI that publishes multi-arch (`linux/amd64`, `linux/arm64`) GHCR images from
  `pixi.lock` (`latest` on `master`; SHA, version, and `stable` on release) and
  attaches a Python wheel to GitHub Releases. Local images: `pixi run docker-build`.
- CI rejects pull requests that do not update `CHANGELOG.md`.
