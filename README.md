# Copynumber-CNV

ccnv is a Python 3.12+ rewrite of the WISECONDOR shallow whole-genome
sequencing copy-number pipeline. `convert` streams BAM/CRAM files into
canonical sample Parquet using Polars-Bio.
`reference` builds autosomal and optional sex-contig models into a versioned
reference Parquet, or a flat uniform-depth model from a FASTA reference when no control
samples are available. `predict` normalizes coverage, segments it with weighted
Binseg, calls gains/losses, and writes 0-based half-open BED (plus optional
SVG). Convert and reference persist Parquet.
CLI commands take `--prefix` and append canonical suffixes themselves.

The rewrite deliberately uses versioned Parquet artifacts instead of legacy
NPZ files. Coordinates are always 0-based, half-open intervals, contig names
and lengths come from the alignment header, and genome layouts are not assumed
to contain 24 chromosomes.

## Installation

```console
python -m pip install .
ccnv --help
```

For a source checkout, install [Pixi](https://pixi.sh) and run `pixi install`.

GitHub Releases attach a Python wheel. Container images are published to
`ghcr.io/matthdsm/copynumber-cnv` for `linux/amd64` and `linux/arm64`:
pushes to `master` tag the commit SHA and `latest`; published releases also
tag the version and `stable`. Local images: `pixi run docker-build`.

```console
docker pull ghcr.io/matthdsm/copynumber-cnv:latest
docker run --rm ghcr.io/matthdsm/copynumber-cnv:latest --help
```

## Pipeline

```console
ccnv convert sample.bam --prefix sample --bin-size 5000

ccnv reference controls/ --prefix reference \
  --autosomal-contig chr1 --autosomal-contig chr2

ccnv reference genome.fa --prefix reference --bin-size 5000

ccnv predict sample.ccnv reference.ccnv \
  --prefix results/sample --ratio-threshold 0.2
```

CRAM conversion also requires `--fasta` and an indexed FASTA (`.fai`).
CLI file inputs must exist: Typer rejects missing BAM/CRAM, FASTA, sample Parquet, and
reference Parquet paths. Missing required indices fail fast with an actionable error. `reference` accepts sample Parquet files and/or
directories; directories are searched recursively for sample Parquet (other
files are ignored). With a FASTA file instead of samples, `reference` writes a
flat uniform-depth model from contig lengths so `predict` can run without a
control cohort. Sex-contig models default to `X`/`Y`/`chrX`/`chrY` and a
`chrY` or `Y` signal contig. Pass `--sex-cutoff` to classify samples by Y
fraction, or omit it to infer a cutoff when groups separate cleanly. Input
samples are rebinned to the target bin size (default 5000) if their bin sizes
differ, provided contig lengths match. See [CLI reference](docs/cli.md)
for every flag. Production cohorts normally need at least ten controls.

`convert` writes `<prefix>.ccnv`. `reference` writes `<prefix>.ccnv`.
Prediction writes:

- `<prefix>_bins.bed`: per-bin ratio, z-score, weight, reference size, mask
  state, and branch, with a header. Coordinates are 0-based half-open genome
  contig names (`id` is `contig:start-end`).
- `<prefix>_segments.bed`: weighted segments with ratio, z-score, weight,
  bin count, and call.
- `<prefix>_calls.bed`: called gains/losses only.
- `<prefix>_plot.svg`: genome-aware plot unless `--no-plot` is passed.
- `<prefix>_bins.ccnv` and `<prefix>_segments.ccnv`: the same evidence
  as typed tables.

Omit `--prefix` on `convert` or `predict` to use the input stem.

## Artifact contracts

Sample Parquet has one row per genomic bin:
`contig:string`, `start:int64`, `end:int64`, and `count:uint64`. Metadata records
schema version, coordinate system, bin size, contig lengths, source/reference
identity, and deterministic QC counters.

Reference Parquet has one row per branch/bin with coordinates, mask state, PCA
mean/loadings, nearest off-contig reference indices/distances, and null ratios.
Metadata records model settings, branch roles, cohort identities, and optional
sex labels/inference. `ccnv.schemas` rejects unsupported versions,
undeclared contigs, duplicate bins, and out-of-bounds intervals before
scientific processing.

The current schema version is `1.0.0`. No compatibility with legacy NPZ files
or numeric chromosome IDs is provided.

## Scientific scope and migration

This is a breaking rewrite, not a drop-in replacement for WisecondorX 1.3.2:

- Weighted Python Binseg replaces R/DNAcopy. Its absolute weighted-SSE penalty
  is not DNAcopy `alpha`, so thresholds require independent validation.
- Sex-contig branches use quality-gated inference from a signal contig
  (`--sex-signal-contig`, optional `--sex-cutoff`), not per-sample labels or
  legacy suffix conventions.
- Coordinates, contig order, and true final-bin ends come from alignment
  headers. There is no human 24-chromosome assumption.
- Compact fixture validation is not clinical validation. Sensitivity,
  specificity, mosaicism limits, GC/library effects, cohort suitability, and
  penalty/call thresholds must be established for each assay.

See [validation evidence](docs/validation.md),
[reproducible benchmarks](docs/benchmarks.md), and the
[CLI reference](docs/cli.md).

## Development

Install [Pixi](https://pixi.sh), then run:

```console
pixi install
pixi run format
pixi run lint
pixi run test
pixi run hooks
```

Inspect the CLI with:

```console
pixi run ccnv --help
```

Changes should be make through pull requests and recorded in [the changelog](CHANGELOG.md)

Baseline outputs captured from WisecondorX v1.3.2 live under
`tests/fixtures/upstream/`. They are immutable scientific comparison data, not
runtime inputs or compatibility promises.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md).
