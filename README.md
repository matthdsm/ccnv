# Copynumber-CNV (`ccnv`)

`ccnv` is an independent Python 3.12+ toolkit for copy-number variation (CNV) detection in shallow whole-genome sequencing (sWGS) data, inspired by the published statistical methodology of WISECONDOR (Straver et al., *Nucleic Acids Research*, 2014) and WisecondorX (Raman et al., *Nucleic Acids Research*, 2019).

- **Versioned Parquet Storage (`1.0.0`)**: Sample bin counts and reference models are stored as Zstandard-compressed Apache Arrow/Parquet `.ccnv` files with embedded QC and provenance metadata.
- **Organism-Agnostic & 0-Based Half-Open Coordinates**: Contig names, order, and lengths are read directly from alignment or FASTA headers without assuming a fixed 24-chromosome layout. All intervals are 0-based half-open `[start, end)`.
- **Pure-Python Execution**: Uses `pysam`, `numpy`, `pyarrow`, `scikit-learn`, and weighted `ruptures` Binary Segmentation.

## Installation

```console
python -m pip install .
ccnv --help
```

For development with [Pixi](https://pixi.sh):

```console
pixi install
pixi run ccnv --help
```

Multi-platform container images (`linux/amd64` and `linux/arm64`) are published to `ghcr.io/matthdsm/copynumber-cnv`:

```console
docker pull ghcr.io/matthdsm/copynumber-cnv:latest
docker run --rm ghcr.io/matthdsm/copynumber-cnv:latest --help
```

## Quickstart

```console
# 1. Convert BAM or CRAM alignments into binned sample Parquet (.ccnv)
ccnv convert sample.bam --prefix sample --bin-size 5000

# 2a. Build a reference model from a cohort of control samples
ccnv reference controls/ --prefix reference --bin-size 5000

# 2b. Or build a flat uniform-depth reference from a FASTA when no controls are available
ccnv reference genome.fa --prefix reference --bin-size 5000

# 3. Inspect reference quality control metrics
ccnv ref-qc reference.ccnv

# 4. Normalize, segment, and call copy-number gains and losses
ccnv predict sample.ccnv reference.ccnv --prefix results/sample
```

- `convert` writes `<prefix>.ccnv` (automatically indexing BAM/CRAM files when needed; CRAM inputs also require `--fasta`).
- `reference` writes `<prefix>.ccnv` and `<prefix>_qc.json` (rebinning input samples in memory when bin sizes differ and contig lengths match).
- `predict` writes bgzipped and tabix-indexed BED and VCFv4.2 tables (`_bins`, `_segments`, and `_calls`) plus an interactive genome-wide `<prefix>_plot.html`.

## Documentation

- [CLI Reference](docs/cli.md): Complete command-line flags, defaults, and I/O specifications.
- [Pipeline Architecture](docs/pipeline.md): Data flow, coordinate contracts, and algorithmic stages.
- [Benchmarks](docs/benchmark.md): Concordance, storage efficiency, and segmentation benchmarks.

## Development

```console
pixi run format
pixi run lint
pixi run test
pixi run hooks
```

All changes must be submitted via pull request and recorded in [CHANGELOG.md](CHANGELOG.md).

## License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md).
