# Copynumber-CNV Agent Guidelines

This document provides project-specific instructions and architectural context for agents working on the Copynumber-CNV (`ccnv`) codebase.

## Project Overview

Copynumber-CNV (`ccnv`) is a bioinformatics toolkit for Copy Number Variation (CNV) analysis in shallow whole-genome sequencing data. It is an independent Python implementation inspired by the published statistical methodology of WISECONDOR (Straver et al., 2014) and WisecondorX (Raman et al., 2019).

### Core Stack

- **Language:** Python 3.12+
- **CLI Framework:** Typer
- **Data Science:** NumPy, SciPy, Polars, Scikit-learn, PyArrow, Ruptures
- **Bioinformatics:** Pysam, pyBigWig
- **Serialization & Validation:** PyYAML, jsonschema
- **Templating:** Jinja2
- **Environment & Dependency Management:** [Pixi](https://pixi.sh). The `default` environment includes the `dev` feature; `prod` is the same runtime dependencies without test/lint tools and is what the Docker image installs.
- **Linting:** Prek & Ruff
- **Testing:** Pytest (using `unittest` style tests)

## Architectural Context

### Pipeline Stages

1. **Convert:** Stream a BAM/CRAM once per contig, creating a pysam index when missing, filter reads, calculate binned counts with NumPy, and write canonical sample Parquet plus deterministic QC metadata. Only canonical chromosomes (1-22, X, Y in UCSC or plain spelling) are binned unless `--all-contigs` is set; mitochondrial, ALT, decoy, random, unplaced, and EBV contigs are skipped. `-t/--threads` is a CPU budget split between contig worker processes and per-handle htslib decode threads; output order follows the header. Sample QC metadata tracks effective reads examined, kept, and filtered strictly across binned target contigs.
2. **Reference:** Ingest cohort samples via a root-list YAML/JSON samplesheet (`-s/--samplesheet`) or build a flat uniform-depth model from a reference FASTA (`--fasta`). Validate sample layouts, rebin sample counts to the target bin size (default 5000 bp) when contig lengths match, normalize depth, optionally load GC content (`--gc`) and mappability BigWig (`--mappability`, where k-mer length should roughly match average sequencing read length) tracks to mask low-mappability/high-N bins (`--min-mappability`, default 0.5) and remove per-sample multiplicative GC/mappability bias, fit deterministic PCA, construct batched off-contig nearest-bin models, and write reference Parquet with reference QC (`<prefix>_qc.json`). Supports `--mode germline | prenatal` (default `germline`). In prenatal mode (`--mode prenatal`), fits 5 kb `chrY` `FFY` (via 1D 2-GMM and non-negative Lasso), coarsens 5 kb counts in memory to 100 kb (`--prenatal-binsize`, default 100000 bp) for `PREFACE` (centered unscaled PCA + L-BFGS-B Sigmoid MLP / OLM + OLS skew correction) and `FFX` (Huber RLM), partitions `female_sex` (diploid `XX` baseline) and `male_sex` branches using prenatal sex classification, and persists native `prenatal_autosomal` (100 kb) and `prenatal_ffy` (5 kb) branches along with `ccnv.prenatal` scalar metadata.
3. **Ref-QC:** Audit an existing reference Parquet file and emit `<prefix>_qc.json` and a stderr usability verdict (`PASS`, `WARN`, `FAIL`), including a prenatal summary block for prenatal references.
4. **Predict:** Project and normalize a sample (rebinning sample counts in memory to reference bin coordinates if bin sizes differ and contig lengths match, leaving reference Parquet untouched), apply stored or overridden (`--gc`, `--mappability`, `--min-mappability`) per-sample GC and mappability normalization before PCA projection, build within-sample references, segment with weighted Ruptures Binseg, call gains/losses, and write bgzipped/indexed bin, segment, and call BED (0-based half-open, extra columns with a `#contig` header), optional VCFv4.2 outputs, interactive Plotly HTML (`_plot.html`, with optional `--cytobands` ideogram track), and sample prediction QC (`<prefix>_qc.json`). Supports `--mode germline | prenatal` (default `germline`) with strict reference mode validation. In prenatal mode, normalizes `chrX` bins against the diploid `XX` (`female_sex`) reference branch, evaluates `FFY`, `FFX`, and `PREFACE`, classifies sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`), selects `active_ff`, annotates segments with expected ratios, implied fetal fractions, and discordance status/reasons (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`), highlights discordant segments in high contrast within the HTML viewer, and records prenatal metrics and warnings in `<prefix>_qc.json`.

A fixed human chromosome layout is intentionally absent; the tool is organism-agnostic.

### Module Structure

- `src/ccnv/cli.py`: The `ccnv` Typer entry point and subcommands (`convert`, `reference`, `ref-qc`, `predict`).
- `src/ccnv/convert.py`: Root orchestrator for the convert stage, multi-process contig streaming, and effective read statistics collection.
- `src/ccnv/reference.py`: Root orchestrator for creating cohort and flat FASTA reference files across germline and prenatal modes.
- `src/ccnv/ref_qc.py`: Root orchestrator for auditing reference Parquet files and emitting usability verdicts.
- `src/ccnv/predict.py`: Root orchestrator for single-sample prediction, within-sample normalization, Ruptures segmentation, calling, and reporting.
- `src/ccnv/models/`: Pure domain types, dataclasses, schemas, configuration containers, and exceptions (`domain`, `config`, `results`, `schemas`, `exceptions`).
- `src/ccnv/io/`: Pure I/O ingestion and persistence routines for alignments, external tracks (GC, mappability, cytobands), samplesheet schemas, and versioned Parquet tables (`alignments`, `tracks`, `samplesheet`, `parquet`).
- `src/ccnv/stats/`: Pure stateless mathematical and machine learning kernels for PCA, batched off-contig nearest-bin search, weighted segmentation costs, regression, and prenatal mathematical estimators (`pca`, `neighbors`, `segmentation`, `regression`, `prenatal_math`).
- `src/ccnv/features/`: In-memory feature transforms, binning, cohort alignment/rebinning, Huber IRLS covariate bias correction, and prenatal reference/inference workflows (`binning`, `cohort`, `covariates`, `prenatal`).
- `src/ccnv/qc/`: Quality control metrics calculation, verification, and JSON serialization for samples, references, and predictions (`sample`, `reference`, `predict`).
- `src/ccnv/report/`: Genomic deliverable generators for bgzipped/tabix BED, VCFv4.2, and interactive Plotly HTML viewer (`bed`, `vcf`, `html`, `templates/genome_plot.html.j2`).

### Storage and Coordinate Contracts

- Schema version `2.0.0`; all internal and BED coordinates are 0-based half-open.
- Sample rows are `contig`, `start`, `end`, and `count`; metadata includes `ccnv.qc` effective read metrics on target contigs.
- Reference rows add `branch`, `masked`, `gc`, `mappability`, `pca_mean`, `pca_loadings`, `nearest_ref_indices`, `mean_distance`, `null_mean`, and `null_std`. Branches include CNV models (`autosomal`, `female_sex`, `male_sex`) and prenatal models (`prenatal_autosomal` at 100 kb, `prenatal_ffy` at 5 kb `chrY`), with scalar prenatal hyperparameters stored under `ccnv.prenatal` metadata.
- Alignment-header contig names, order, lengths, and partial final bins are authoritative. Readers reject incompatible versions, layouts, and bounds.

## Development Workflow

### CI Publish

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and hooks on every push and pull request. Pull requests must change `CHANGELOG.md`. After those checks pass:

- A push to `master` builds the Docker image and pushes it to GHCR tagged with the commit SHA and `latest`.
- A published GitHub Release builds a Python wheel, attaches it to that release, and pushes the image tagged with the commit SHA, the version (release tag with a leading `v` stripped), and `stable`.

Images live at `ghcr.io/<owner>/<repo>` (currently `ghcr.io/matthdsm/copynumber-cnv`) as a multi-platform manifest for `linux/amd64` and `linux/arm64`. The Dockerfile installs the `prod` Pixi environment with `pixi install --locked` so the image matches `pixi.lock`. The image `ENTRYPOINT` is `ccnv`.

### Pixi Tasks

Use `pixi run` for all standard development tasks to ensure environment consistency:

- `pixi run lint`: Check code quality with Ruff.
- `pixi run format`: Format code with Ruff.
- `pixi run test`: Run the test suite.
- `pixi run hooks`: Run pre-commit hooks (managed by `prek`).
- `pixi run docker-build`: Build the Docker image for `linux/amd64` and `linux/arm64` (Buildx cache; not loaded into the local daemon).

### Code Style

- Adhere to the Ruff configuration in `pyproject.toml`.
- **Import Sorting:** Ruff handles this; ensure it is run before committing.
- **Typing:** Use maximal type annotations.

### Code Changes and Versioning

- **Code Changes:** All changes should be made through a pull request. Each pull request needs to be recorded in `CHANGELOG.md`, following best practices found [here](https://keepachangelog.com/en/1.1.0/#how).
- **Versioning:** Follow semantic versioning for releases, updating the version in `pyproject.toml` and `src/ccnv/__init__.py` accordingly.

### Testing Guidelines

- **Contract Tests:** When adding features that change how data is stored or processed, update the contract tests in `tests/`.
- **Functional Tests:** Aim for high coverage of the computational logic.
- **Pipeline Tests:** Keep BAM and CRAM end-to-end coverage in `tests/test_pipeline.py`.

## Agent Specific Instructions

- **Environment Awareness:** Always prefer `pixi run <command>` over direct execution if a pixi task exists.
- **Surgical Updates:** When fixing bugs, prioritize small, targeted changes and adding a corresponding test case.
- **Documentation:** Update this document and README if you make changes that affect the overall architecture or development workflow. Keep all WisecondorX comparisons strictly inside `docs/benchmark.md`.
- **Markdown Formatting:** Do not hard-wrap lines in Markdown files; rely on editor soft-wrapping for line length. Keep paragraphs and list items on a single logical line. Avoid unnecessary newlines, redundant blank lines, and empty lines inside Mermaid diagram blocks.
