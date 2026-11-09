# `ccnv.report`

## Overview

The `ccnv.report` package generates downstream bioinformatic deliverables, bgzipped/tabix-indexed genomic tables, and interactive visualization reports. It formats binned evidence, segmented intervals, and copy-number calls into compressed BED and VCFv4.2 specifications, and renders self-contained Plotly HTML genome viewers using Jinja2 templating with optional cytoband ideograms and prenatal discordance overlays.

## Interface

- `[src/ccnv/report/__init__.py:write_outputs]`: Coordinates the serialization of all prediction artifacts (`.bed.gz`, `.vcf.gz`, `.html`).
- `[src/ccnv/report/bed.py:write_compressed_bed]`: Writes bgzip-compressed BED tables with `#contig` headers and creates companion tabix (`.tbi`) index files.
- `[src/ccnv/report/bed.py:bin_bed_rows]`: Formats binned ratio, observed count, expected depth, and Z-score records into BED lines.
- `[src/ccnv/report/bed.py:segment_bed_rows]`: Formats segmented genomic intervals and prenatal discordance annotations into BED lines.
- `[src/ccnv/report/bed.py:calls_bed_rows]`: Formats called copy-number gains and losses into BED lines.
- `[src/ccnv/report/bed.py:sort_bin_results]`, `[src/ccnv/report/bed.py:sort_segments]`: Sorts genomic intervals across contigs using deterministic natural karyotype ordering.
- `[src/ccnv/report/vcf.py:build_vcf_header]`: Assembles VCFv4.2 headers defining contig lengths, INFO tags, and FORMAT definitions.
- `[src/ccnv/report/vcf.py:write_bins_vcf]`: Serializes binned observations into bgzipped and tabix-indexed VCFv4.2 files.
- `[src/ccnv/report/vcf.py:write_segments_vcf]`: Serializes segments and calls into bgzipped and tabix-indexed VCFv4.2 files.
- `[src/ccnv/report/html.py:render_genome_plot]`: Compiles an interactive Plotly HTML viewer from `templates/genome_plot.html.j2`.
- `[src/ccnv/report/html.py:write_genome_plot]`: Writes the rendered HTML string to disk.

## Invariants

- `[src/ccnv/report/bed.py:write_compressed_bed]`: Emits 0-based half-open coordinates with `#contig` header; compresses with bgzip and indexes with tabix.
- `[src/ccnv/report/vcf.py:build_vcf_header]`: Produces standard VCFv4.2 headers with matching contig lengths and caller metadata.
- `[src/ccnv/report/html.py:render_genome_plot]`: Renders self-contained Plotly HTML via Jinja2 template (`templates/genome_plot.html.j2`), highlighting discordant segments in prenatal mode.

## Side Effects

- `[src/ccnv/report/bed.py:write_compressed_bed]`: Writes bgzipped BED files (`.bed.gz`) and creates tabix indices (`.tbi`) via `pysam`.
- `[src/ccnv/report/vcf.py:write_bins_vcf]`, `[src/ccnv/report/vcf.py:write_segments_vcf]`: Writes bgzipped VCF files (`.vcf.gz`) and creates tabix indices (`.tbi`) via `pysam`.
- `[src/ccnv/report/html.py:render_genome_plot]`: Writes standalone HTML visualization `<prefix>_plot.html` to disk.

## Verification

- Run: `pytest`
