# `ccnv.report`

## Overview

The `ccnv.report` package generates downstream bioinformatic deliverables for `ccnv predict`: bgzipped and tabix-indexed 0-based half-open BED tables, bgzipped and tabix-indexed VCFv4.2 structural variant files, and interactive Plotly HTML genome viewers (`templates/genome_plot.html.j2`) with optional cytoband ideograms and prenatal/fragmentomic segment overlays.

## Interface

- [`__init__.py:write_outputs`](./__init__.py): Coordinates writing `_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz` (plus `.tbi` indices), optional VCFv4.2 files (`_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`), and optional `<prefix>_plot.html`.
- [`bed.py:write_compressed_bed`](./bed.py) & [`format_bed_float`](./bed.py): Write bgzipped BED tables with `#contig` headers and build `.tbi` tabix indices via `pysam`.
- [`bed.py:bin_bed_rows`](./bed.py): Formats bin rows under `BINS_BED_HEADER` (`#contig`, `start`, `end`, `id`, `ratio`, `zscore`, `weight`, `reference_size`, `masked`, `branch`).
- [`bed.py:segment_bed_rows`](./bed.py), [`calls_bed_rows`](./bed.py), & [`prenatal_bed_fields`](./bed.py): Format segment and gain/loss call rows, appending prenatal and fragmentomic columns (`expected_ratio`, `implied_ff`, `discordant`, `discordance_reason`, `frag_short_ratio`, `frag_llr_size`, `origin_call`) when `mode == "prenatal"`.
- [`bed.py:sort_bin_result`](./bed.py), [`sort_bin_results`](./bed.py), & [`sort_segments`](./bed.py): Deterministically sort bins and segments in natural chromosome order (`natural_sort_key`).
- [`vcf.py:build_vcf_header`](./vcf.py), [`write_bins_vcf`](./vcf.py), [`write_segments_vcf`](./vcf.py), & [`write_calls_vcf`](./vcf.py): Serialize bgzipped/tabix-indexed VCFv4.2 deliverables with `<DUP>`/`<DEL>` symbolic alleles, standard `INFO`/`FORMAT` tags, and prenatal extensions (`##fetalSex`, `##fetalFraction`, `##scaCall`, `EXP_LOG2R`, `IMPLIED_FF`, `DISCORDANT`, `DISC_REASON`, `FRAG_SHORT_R`, `FRAG_LLR`, `ORIGIN`, `FF`).
- [`html.py:render_genome_plot`](./html.py) & [`write_genome_plot`](./html.py): Render the interactive Plotly WebGL genome viewer string from `templates/genome_plot.html.j2` (referencing the Plotly.js CDN) and write `<prefix>_plot.html` to disk.

## Invariants

- [`bed.py:write_compressed_bed`](./bed.py): Emits 0-based half-open intervals sorted in natural contig order (`preset="bed"` tabix indexing).
- [`vcf.py:write_segments_vcf`](./vcf.py) & [`write_calls_vcf`](./vcf.py): Emit 1-based VCF `POS` (`start + 1`) and `END` (`end`) coordinates sorted in natural contig order (`preset="vcf"` tabix indexing), with `ORIGIN` restricted to `FETAL_CONFIRMED`, `MATERNAL_CONFIRMED`, or `INDETERMINATE` when non-null.
- [`html.py:render_genome_plot`](./html.py): Pure in-memory template renderer returning an HTML string without filesystem writes.

## Side Effects

- [`bed.py:write_compressed_bed`](./bed.py): Writes `.bed.gz` and `.bed.gz.tbi` files to disk via `pysam`.
- [`vcf.py:write_bins_vcf`](./vcf.py), [`write_segments_vcf`](./vcf.py), & [`write_calls_vcf`](./vcf.py): Write `.vcf.gz` and `.vcf.gz.tbi` files to disk via `pysam`.
- [`html.py:write_genome_plot`](./html.py): Creates parent directories and writes `<prefix>_plot.html` to disk.

## Verification

- Run: `pixi run test`
