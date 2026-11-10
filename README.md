# Copynumber-CNV (`ccnv`)

`ccnv` is an independent Python 3.14+ bioinformatics toolkit for copy-number variation (CNV) detection in shallow whole-genome sequencing (sWGS) data, building on published within-sample reference normalization methodology (Straver et al., *Nucleic Acids Research*, 2014; Raman et al., *Nucleic Acids Research*, 2019).

> [!NOTE]
> Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

Complete documentation, architecture diagrams, and methodology are available at [https://matthdsm.github.io/copynumber-cnv/](https://matthdsm.github.io/copynumber-cnv/).

---

## Features

- **Versioned Parquet Storage (`1.0.0`)**: Sample bin counts, per-bin cfDNA fragmentomic channels (`fragment_lengths`, `endpoint_count`, `frag_gc_sum`), and reference models stored as Zstandard-compressed Apache Arrow/Parquet `.ccnv` files with embedded QC, `ccnv.fragmentomics`, and optional `ccnv.prenatal` metadata.
- **Germline & Prenatal Analysis Modes**: Supports constitutional CNV profiling (`--mode germline`) and cfDNA non-invasive prenatal screening (`--mode prenatal`) with pure NumPy/SciPy/scikit-learn fetal fraction estimation (`FFY`, `FFX`, `FFA`, `FF_size`), depth-conserving Fisher-optimal fragment length read weighting, 2D stoichiometric Bayesian sex chromosome aneuploidy calling, segment discordance annotation, and fragmentomic origin calling.
- **Organism-Agnostic & 0-Based Coordinates**: Contig names, order, and lengths are read directly from alignment or FASTA headers without assuming a fixed 24-chromosome layout. All intervals are 0-based half-open `[start, end)`.
- **Pure-Python & DataFrame-Native**: Built on `polars`, `pysam`, `pyBigWig`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, and weighted `ruptures` segmentation without external R or container runtime dependencies.

---

## Installation

### pip

```console
python -m pip install .
ccnv --help
```

### Pixi

```console
pixi install
pixi run ccnv --help
```

### Docker

Multi-platform container images (`linux/amd64` and `linux/arm64`) are published to GHCR:

```console
docker pull ghcr.io/matthdsm/copynumber-cnv:latest
docker run --rm ghcr.io/matthdsm/copynumber-cnv:latest ccnv --help
```

---

## Quickstart

```console
# 1. Extract BAM or CRAM alignments into binned sample Parquet (.ccnv)
ccnv extract sample.bam --prefix sample --bin-size 5000

# 2a. Build a germline reference model from a cohort samplesheet
ccnv reference -s cohort.yaml --prefix reference --bin-size 5000 --gc genome.fa --mappability mappability.bw

# 2b. Or build a flat uniform-depth germline reference from a FASTA
ccnv reference --fasta genome.fa --prefix reference --bin-size 5000 --mappability mappability.bw

# 2c. Or build a unified prenatal reference from a maternal cfDNA cohort samplesheet
ccnv reference -s cohort.yaml --mode prenatal --prefix prenatal_ref --bin-size 5000

# 3. Audit reference quality control metrics
ccnv ref-qc reference.ccnv

# 4a. Predict copy-number gains and losses from .ccnv, .bam, or .cram (with --fasta)
ccnv predict sample.ccnv reference.ccnv --prefix results/sample

# 4b. Or run prenatal prediction with fetal fraction estimation and sex chromosome analysis
ccnv predict sample.ccnv prenatal_ref.ccnv --mode prenatal --prefix results/sample_prenatal
```

Outputs include bgzipped/indexed BED tables, VCFv4.2 files, interactive Plotly HTML reports (`_plot.html`), and structured JSON metrics (`<prefix>_qc.json`).

---

## Documentation

Visit the [documentation site](https://matthdsm.github.io/copynumber-cnv/) for detailed guides:

- [Pipeline Architecture](https://matthdsm.github.io/copynumber-cnv/pipeline/): 4-stage execution model and storage contracts.
- [CLI Reference](https://matthdsm.github.io/copynumber-cnv/cli/): Flag specification, inputs, outputs, and exit codes.
- [Mathematical Methodology](https://matthdsm.github.io/copynumber-cnv/methodology/): Statistical formulations, models, and citations.

---

## Development & Verification

```console
pixi run format      # Format code with Ruff
pixi run lint        # Lint code with Ruff
pixi run test        # Run unit & contract test suite
pixi run hooks       # Run pre-commit hooks with Prek
pixi run docs-build  # Build documentation site with Zensical
pixi run docs-serve  # Serve documentation site locally
```

All changes must be submitted via pull request and recorded in [CHANGELOG.md](CHANGELOG.md).

---

## License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md).
