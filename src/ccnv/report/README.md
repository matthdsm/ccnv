# `ccnv.report`

## Overview

The `ccnv.report` package generates downstream bioinformatic deliverables for `ccnv predict`: bgzipped and tabix/CSI-indexed 0-based half-open BED tables, bgzipped and tabix/CSI-indexed VCFv4.2 structural variant files, and interactive Plotly HTML genome viewers (`templates/genome_plot.html.j2`) with optional cytoband ideograms and prenatal/fragmentomic segment overlays. All BGZF, index, and HTML writes execute atomically via sibling temporary files, with single-pass BGZF compression and indexing handled by the compiled Rust `ccnv._core` writer.

## Interface

- [`__init__.py:write_outputs`](./__init__.py): Unpacks sorted `SegmentResult` records into a shared Polars DataFrame once and coordinates atomic streaming of `_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz` (plus `.tbi`/`.csi` indices), optional VCFv4.2 files (`_segments.vcf.gz`, `_calls.vcf.gz`, and opt-in `_bins.vcf.gz`), and optional `<prefix>_plot.html`.
- [`_common.py:ContigBinChunk`](./_common.py), [`iter_contig_bin_chunks`](./_common.py), [`bin_chunk_to_dataframe`](./_common.py), [`segments_to_dataframe`](./_common.py), [`resolve_segments_dataframe`](./_common.py), [`sort_segments_dataframe`](./_common.py), [`max_bin_end`](./_common.py), [`max_df_end`](./_common.py), [`sort_bin_result`](./_common.py), [`sort_bin_results`](./_common.py), [`sort_segments`](./_common.py), & [`write_bgzf_atomic`](./_common.py): Shared natural chromosome ordering, per-contig bin chunk partitioning, coordinate upper-bound helpers, vectorized Polars bin/segment DataFrame builders and sorters (mapping non-finite floats to `null`), and atomic BGZF + `.tbi`/`.csi` replacement (unlinking stale opposite index extensions).
- [`bed.py:write_bins_bed_streaming`](./bed.py) & [`write_segments_bed_streaming`](./bed.py): Format and stream `#contig`-headed BED tables (`BINS_BED_HEADER`, `SEGMENTS_BED_HEADER`, `CALLS_BED_HEADER`, `PRENATAL_SEGMENTS_BED_HEADER`, `PRENATAL_CALLS_BED_HEADER`, supporting `calls_only=True` and pre-built `segments_df`) into single-pass BGZF + `.tbi`/`.csi` files via `ccnv._core`.
- [`vcf.py:build_vcf_header`](./vcf.py), [`write_bins_vcf`](./vcf.py), [`write_segments_vcf`](./vcf.py), & [`write_vcf_streaming`](./vcf.py): Serialize bgzipped/`.tbi`/`.csi`-indexed VCFv4.2 segment, call (`calls_only=True`), and opt-in bin deliverables with `<DUP>`/`<DEL>`/`<CNV>` symbolic alleles, standard `INFO`/`FORMAT` tags, and prenatal extensions (`##fetalSex`, `##fetalFraction`, `##scaCall`, `EXP_LOG2R`, `IMPLIED_FF`, `DISCORDANT`, `DISC_REASON`, `FRAG_SHORT_R`, `FRAG_LLR`, `ORIGIN`, `MOSAIC_FRAC`).
- [`html.py:render_genome_plot`](./html.py), [`decimate_bins_m4`](./html.py), & [`encode_b64`](./html.py): Decimate dense bin profiles using outlier-preserving M4 bucket selection, render the cached Jinja2 template `templates/genome_plot.html.j2` (referencing `PLOTLY_CDN_URL`), and atomically write `<prefix>_plot.html` to disk.

## Invariants

- [`bed.py:write_bins_bed_streaming`](./bed.py) & [`write_segments_bed_streaming`](./bed.py): Emit 0-based half-open intervals sorted in natural contig order (`preset="bed"` indexing), formatting missing/non-finite bin values as `"."` and optional empty segment columns as `""`.
- [`vcf.py:write_bins_vcf`](./vcf.py) & [`write_segments_vcf`](./vcf.py): Emit 1-based VCF `POS` (`start + 1`) and `END` (`end`) coordinates sorted in header/natural contig order (`preset="vcf"` indexing), with `ORIGIN` reporting 5-state origin classifications (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$).
- [`html.py:decimate_bins_m4`](./html.py): Preserves sorted unique bucket extrema (`first`, `argmin`, `argmax`, `last`) plus all finite $`\vert z \vert \ge z_{\text{preserve}}`$ outlier bins.

## Side Effects

- [`bed.py:write_bins_bed_streaming`](./bed.py) & [`write_segments_bed_streaming`](./bed.py): Atomically write `.bed.gz` and `.tbi`/`.csi` index files to disk via `write_bgzf_atomic`.
- [`vcf.py:write_bins_vcf`](./vcf.py), [`write_segments_vcf`](./vcf.py), & [`write_vcf_streaming`](./vcf.py): Atomically write `.vcf.gz` and `.tbi`/`.csi` index files to disk via `write_bgzf_atomic`.
- [`html.py:render_genome_plot`](./html.py): Creates parent directories and atomically writes `<prefix>_plot.html` to disk via sibling temporary file replacement.

## Verification

- Run: `pixi run test`
