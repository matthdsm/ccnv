# `ccnv.templates`

## Overview

`ccnv.templates` packages the Jinja2 HTML template resources used by `ccnv.report` to render interactive genome-wide and per-chromosome Plotly copy-number visualizations.

## Interface

- `[src/ccnv/templates/__init__.py:__doc__]`: Package marker enabling `importlib.resources.files("ccnv.templates")` lookup for bundled template assets.
- `[src/ccnv/templates/genome_plot.html.j2:title]`: Jinja2 template variable setting the HTML document title and header sample label.
- `[src/ccnv/templates/genome_plot.html.j2:plotly_cdn_url]`: Jinja2 template variable specifying the external Plotly.js script URL.
- `[src/ccnv/templates/genome_plot.html.j2:payload_json]`: Jinja2 template variable injecting the serialized JSON payload (`D`) containing contig metadata, base64-encoded typed bin arrays, segments, cytobands, and call thresholds.

## Invariants

- `[src/ccnv/templates/genome_plot.html.j2:D.bins.c]`: Decoded in browser from base64 into a little-endian `Uint16Array` of contig indices.
- `[src/ccnv/templates/genome_plot.html.j2:D.bins.s]`: Decoded in browser from base64 into a little-endian `Int32Array` of 0-based bin start coordinates.
- `[src/ccnv/templates/genome_plot.html.j2:D.bins.e]`: Decoded in browser from base64 into a little-endian `Int32Array` of 0-based half-open bin end coordinates.
- `[src/ccnv/templates/genome_plot.html.j2:D.bins.r]`: Decoded in browser from base64 into a little-endian `Float32Array` of bin log2 ratios.
- `[src/ccnv/templates/genome_plot.html.j2:D.bins.z]`: Decoded in browser from base64 into a little-endian `Float32Array` of bin Z-scores.
- `[src/ccnv/templates/genome_plot.html.j2:totalLen]`: Clamped to `>= 1` when total contig length is `<= 0` to prevent division-by-zero during genome-wide coordinate scaling.
- `[src/ccnv/templates/genome_plot.html.j2:classify]`: Classifies a segment only when `Math.abs(z) >= zT && Math.abs(r) >= rT`, returning `1` for gains (`r > 0 && z >= 0`), `-1` for losses (`r < 0 && z <= 0`), and `0` otherwise.
- `[src/ccnv/templates/genome_plot.html.j2:#z-input]`: Numeric threshold input constrained to `min="0"` and `step="0.1"`.
- `[src/ccnv/templates/genome_plot.html.j2:#r-input]`: Numeric threshold input constrained to `min="0"` and `step="0.05"`.

## Side Effects

- `[src/ccnv/templates/genome_plot.html.j2:plotly_cdn_url]`: Fetches the external Plotly.js bundle over the network via `<script src="{{ plotly_cdn_url }}">` when opened in a browser.
- `[src/ccnv/templates/genome_plot.html.j2:#z-input]`: Mutates the DOM input element value to `D.z_thresh` on page initialization and re-renders Plotly traces and summary tables on input change.
- `[src/ccnv/templates/genome_plot.html.j2:#r-input]`: Mutates the DOM input element value to `D.r_thresh` on page initialization and re-renders Plotly traces and summary tables on input change.

## Verification

- Run: `pytest`
