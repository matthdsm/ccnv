# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ccnv convert`: stream BAM/CRAM into versioned sample Parquet, auto-index
  missing `.bai`/`.crai`, bin canonical chromosomes (or `--all-contigs`), and
  split `--threads` across contig workers and htslib decode.
- `ccnv reference`: build autosomal and optional sex-contig models from control
  Parquet, or a flat uniform-depth model from `--fasta` when there is no cohort.
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
