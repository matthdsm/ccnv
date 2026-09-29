# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Prediction BED outputs (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`) are now always bgzip-compressed and tabix-indexed (`.tbi`), with `#contig` headers for standard tabix/IGV compatibility.
- Prediction outputs (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`, `_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`, `_bins.ccnv`, `_segments.ccnv`, and `_plot.svg`) now sort chromosomes and intervals in natural sort order (e.g. `chr1`, `chr2`, ..., `chr10`, ..., `chrX`, `chrY`) instead of lexicographical order.
- Parallelized `ccnv reference` execution controlled by a `-t/--threads` flag (defaulting to system CPU count), multi-processing cohort sample loading and rebinning via `src/ccnv/cohort.py`, parallelizing batched off-contig nearest-bin selection via `src/ccnv/reference_compute.py`, and multi-processing null model derivation.
- Extracted cohort ingestion and validation into `src/ccnv/cohort.py` and compute kernels into `src/ccnv/reference_compute.py`, keeping modules modular and strictly below 1,000 lines.
- Replaced boolean array masking with precomputed contiguous contig slice bounds in `select_nearest_bins`, accelerating per-query candidate masking by over 400x.
- Vectorized `rebin_sample_table` contig boundaries using NumPy run-length transitions instead of converting Arrow contig columns to Python string lists.
- Optimized reference schema (`SCHEMA_VERSION = "1.0.0"`), replacing variable-length `null_ratios` and `nearest_ref_distances` with pre-filtered sorted `nearest_ref_indices`, scalar `mean_distance`, `null_mean`, and `null_std`.
- Reference Parquet writing now uses Zstandard (level 3) compression, dictionary encoding on categorical strings (`branch`, `contig`), and Delta Binary Packing on coordinates (`start`, `end`) and neighbor indices (`nearest_ref_indices.list.element`), shrinking reference files by ~6.5x (e.g., from 2.16 GB down to 333 MB).
- Vectorized `predict.py`, `convert.py`, `reference_qc.py`, and `schemas.py` using PyArrow chunk buffer slicing and NumPy vectorized ops, slashing reference reading, bounds checking, and normalization time by over 5x.
- Unified default bin size to 5,000 bp across `convert`, `reference`, and models.
- `ccnv reference` accepts FASTA inputs as a positional argument alongside sample Parquet inputs, removing the `--fasta` option.
- `ccnv reference` rebins sample counts to `--bin-size` when cohort samples differ in native bin size, requiring matching contig lengths.
- `ccnv predict` rebins sample counts in memory to match the reference bin coordinates when bin sizes differ, without modifying the reference file.
- Convert and FASTA reference now share one contig-layout selector (`select_layout`) so canonical filtering is no longer duplicated, and `GenomeProfile` is the in-memory layout type. Prediction I/O lives in `ccnv.report`, with result types in `ccnv.results`. Bin coordinates are read through `schemas.bin_coordinates`.
- CLI tests compare Rich output after stripping ANSI so CI color does not split flags such as ``--loglevel``.

### Fixed

- Resolved `UnboundLocalError` for `fasta` variable in `ccnv reference` CLI command.

### Added

- VCFv4.2 prediction outputs: `ccnv predict` now generates bgzip-compressed and tabix-indexed VCF files for bins (`_bins.vcf.gz`), segments (`_segments.vcf.gz`), and called events (`_calls.vcf.gz`) using native `pysam.VariantFile` and `pysam.BGZFile` single-pass streaming, containing standard symbolic SV tags (`<DUP>`, `<DEL>`, `<CNV>`), 1-based intervals, and copy ratio / z-score format metrics, enabled by default with `--vcf / --no-vcf` control.
- `ccnv ref-qc`: evaluate reference model usability and WisecondorX-derived quality heuristics (`PASS`, `WARN`, `FAIL`) on existing reference files, emitting `<prefix>_qc.json` and a stderr summary.

- Reference creation quality control: `ccnv reference` automatically evaluates reference metrics and writes `<prefix>_qc.json` during build (toggleable via `--qc-report / --no-qc-report`).
- Comprehensive pipeline architecture documentation (`docs/pipeline.md`) with detailed Mermaid flowcharts for all pipeline subcommands (`convert`, `reference`, `ref-qc`, `predict`), and embedded stage flowcharts in the CLI reference (`docs/cli.md`).
- `rebin_counts` and `rebin_sample_table` for 1D mass-conserving count interpolation across bin grids.
- `ccnv convert`: stream BAM/CRAM into versioned sample Parquet, auto-index missing `.bai`/`.crai`, bin canonical chromosomes (or `--all-contigs`), and split `--threads` across contig workers and htslib decode.
- `ccnv reference`: build autosomal and optional sex-contig models from control Parquet, or a flat uniform-depth model from FASTA when there is no cohort.
- `ccnv predict`: PCA-normalize, segment with weighted Binseg, call gains and losses, and write 0-based half-open BED, optional SVG, and matching Parquet.
- Versioned Parquet contracts (`1.0.0`), header-authoritative contig layouts, and organism-agnostic naming (no fixed 24-chromosome genome).
- Sex-contig branches inferred from a Y-signal contig (`--sex-cutoff` optional); no per-sample sex labels on the CLI.
- CI that publishes multi-arch (`linux/amd64`, `linux/arm64`) GHCR images from `pixi.lock` (`latest` on `master`; SHA, version, and `stable` on release) and attaches a Python wheel to GitHub Releases. Local images: `pixi run docker-build`.
- Real-world NIPT conversion validation and storage comparison against WisecondorX documented in `docs/validation.md`.
- Reference creation comparison against WisecondorX across 8 pipeline stages, numerical concordance, and 9.2x storage footprint reduction documented in `docs/validation.md`.
- Predict stage comparison against WisecondorX across 8 pipeline stages, bin concordance (r=0.9869), weighted Ruptures segmentation, and 4.2x Parquet compression advantage documented in `docs/validation.md`.
- CI rejects pull requests that do not update `CHANGELOG.md`.



