# `ccnv`

`ccnv` is an independent Python 3.14+ bioinformatics toolkit for copy-number variation (CNV) detection in whole-genome sequencing (WGS) data at any coverage, building on published within-sample reference normalization methodology (Straver et al., *Nucleic Acids Research*, 2014; Raman et al., *Nucleic Acids Research*, 2019).

> [!NOTE]
> `ccnv` is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

Complete documentation, architecture diagrams, and methodology are available at [https://matthdsm.github.io/ccnv/](https://matthdsm.github.io/ccnv/).

---

## Features

- **Versioned Parquet Storage (`2.0.0`)**: Sample sketches, per-bin cfDNA fragmentomic channels (`count`, `paired_count`, `frag_len_sum`, `short_frag_count`, `endpoint_count`, `frag_gc_sum`, `frag_bands`), and reference prototype models are stored as Zstandard-compressed Apache Arrow/Parquet `.ccnv` files with embedded QC, `ccnv.fragmentomics`, and optional `ccnv.prenatal` metadata.
- **Germline & Prenatal Analysis Modes**: Supports constitutional CNV profiling (`--mode germline`) and cfDNA non-invasive prenatal screening (`--mode prenatal`) with closed-form Poisson MSY `FFY`, convex logit-link `FFA`, forward $`R_X`$, reference-side 24-band Fisher weighting, 4D erfc ray-marginalized SCA classification, Student-$`t_4`$ active FF fusion, and 5-state self-referenced origin calling.
- **Calibrated Pelt Segmentation & Integer CN Bayes Factors**: Calibrated `ruptures.Pelt` with mBIC penalty and dyadic multi-scale BLUE Z-scores, paired with discrete integer copy-number log-Bayes factors ($`CN \in \lbrace 0, 1, 2, 3, 4+\rbrace`$) recovering homozygous deletions ($`CN=0`$) and focal deletions down to $`0.1\times`$ sequencing coverage.
- **Rust-Accelerated & Organism-Agnostic Execution**: Built on compiled Rust `ccnv._core` (`noodles` and `bigtools`), `polars`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, and `ruptures` without runtime dependencies on `pysam` or `pybigwig`. Contig names, order, and lengths are read directly from alignment or FASTA headers without assuming a fixed 24-chromosome layout. All intervals are 0-based half-open `[start, end)`.

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
docker pull ghcr.io/matthdsm/ccnv:latest
docker run --rm ghcr.io/matthdsm/ccnv:latest ccnv --help
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

Visit the [documentation site](https://matthdsm.github.io/ccnv/) for detailed guides:

- [Pipeline Architecture](https://matthdsm.github.io/ccnv/pipeline/): 4-stage execution model and storage contracts.
- [CLI Reference](https://matthdsm.github.io/ccnv/cli/): Flag specification, inputs, outputs, and exit codes.
- [API Reference](https://matthdsm.github.io/ccnv/api/): Module-by-module function reference and call-graph DAGs.
- [Mathematical Methodology](https://matthdsm.github.io/ccnv/methodology/): Statistical formulations, models, and citations.

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

Licensed under the GNU Affero General Public License v3 or later (AGPL-3.0-or-later). See [LICENSE.md](LICENSE.md).
