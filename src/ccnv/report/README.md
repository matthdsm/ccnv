# `ccnv.report`

## Overview

The `ccnv.report` package generates downstream bioinformatic deliverables for `ccnv predict`: bgzipped and tabix/CSI-indexed 0-based half-open BED tables, bgzipped and tabix/CSI-indexed VCFv4.2 structural variant files, and interactive Plotly HTML genome viewers (`templates/genome_plot.html.j2`) with optional cytoband ideograms and prenatal/fragmentomic segment overlays. All BGZF and index writes execute in a single pass via the compiled Rust `ccnv._core` writer.

## Interface

- [`__init__.py:write_outputs`](./__init__.py): Coordinates single-pass streaming of `_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz` (plus `.tbi`/`.csi` indices), optional VCFv4.2 files (`_segments.vcf.gz`, `_calls.vcf.gz`, and opt-in `_bins.vcf.gz`), and optional `<prefix>_plot.html`.
- [`_common.py:ContigBinChunk`](./_common.py), [`sort_bin_result`](./_common.py), [`sort_bin_results`](./_common.py), [`sort_segments`](./_common.py), & [`segments_to_report_df`](./_common.py): Shared natural chromosome ordering, per-contig bin chunk partitioning, and vectorized Polars segment DataFrame construction across BED and VCF writers.
- [`bed.py:write_bins_bed_streaming`](./bed.py), [`write_segments_bed_streaming`](./bed.py), & [`write_calls_bed_streaming`](./bed.py): Format and stream `#contig`-headed BED tables (`BINS_BED_HEADER`, `SEGMENTS_BED_HEADER`, `PRENATAL_SEGMENTS_BED_HEADER`) into single-pass BGZF + `.tbi`/`.csi` files via `ccnv._core`.
- [`vcf.py:build_vcf_header`](./vcf.py), [`write_bins_vcf_streaming`](./vcf.py), & [`write_segments_vcf_streaming`](./vcf.py): Serialize bgzipped/`.tbi`/`.csi`-indexed VCFv4.2 segment, call, and opt-in bin deliverables with `<DUP>`/`<DEL>` symbolic alleles, standard `INFO`/`FORMAT` tags, and prenatal extensions (`##fetalSex`, `##fetalFraction`, `##scaCall`, `EXP_LOG2R`, `IMPLIED_FF`, `DISCORDANT`, `DISC_REASON`, `FRAG_SHORT_R`, `FRAG_LLR`, `ORIGIN`, `MOSAIC_FRAC`, `FF`).
- [`html.py:render_genome_plot`](./html.py) & [`write_genome_plot_streaming`](./html.py): Render the interactive Plotly WebGL genome viewer from `templates/genome_plot.html.j2` (referencing the Plotly.js CDN) and write `<prefix>_plot.html` to disk.

## Invariants

- [`bed.py:write_bins_bed_streaming`](./bed.py) & [`write_segments_bed_streaming`](./bed.py): Emit 0-based half-open intervals sorted in natural contig order (`preset="bed"` indexing).
- [`vcf.py:write_segments_vcf_streaming`](./vcf.py): Emits 1-based VCF `POS` (`start + 1`) and `END` (`end`) coordinates sorted in natural contig order (`preset="vcf"` indexing), with `ORIGIN` reporting 5-state origin classifications (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$).
- [`html.py:render_genome_plot`](./html.py): Pure in-memory template renderer returning an HTML string without filesystem writes.

## Side Effects

- [`bed.py:write_bins_bed_streaming`](./bed.py), [`write_segments_bed_streaming`](./bed.py), & [`write_calls_bed_streaming`](./bed.py): Write `.bed.gz` and index files to disk via `ccnv._core`.
- [`vcf.py:write_bins_vcf_streaming`](./vcf.py) & [`write_segments_vcf_streaming`](./vcf.py): Write `.vcf.gz` and index files to disk via `ccnv._core`.
- [`html.py:write_genome_plot_streaming`](./html.py): Creates parent directories and writes `<prefix>_plot.html` to disk.

## Verification

- Run: `pixi run test`
