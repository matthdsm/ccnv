# Copynumber-CNV Agent Guidelines

This document provides project-specific instructions and architectural context for agents when working on the Copynumber-CNV (ccnv for short) codebase.

## Project Overview

Copynumber-CNV is a bioinformatics tool for Copy Number Variation (CNV) analysis in whole-genome sequencing data. It is a next generation version of the algorithms used by WISECONDOR(X).

### Core Stack

- **Language:** Python 3.12+
- **CLI Framework:** Typer
- **Data Science:** NumPy, SciPy, Polars, Scikit-learn, PyArrow, Ruptures
- **Bioinformatics:** Pysam
- **Environment & Dependency Management:** [Pixi](https://pixi.sh). The `default`
  environment includes the `dev` feature; `prod` is the same runtime
  dependencies without test/lint tools and is what the Docker image installs.
- **Linting**: Prek & Ruff
- **Testing:** Pytest (using `unittest` style tests)

## Architectural Context

### Pipeline Stages

1. **Convert:** Stream a BAM/CRAM once per contig, creating a pysam index when
   missing, filter reads, calculate binned counts with NumPy, and write
   canonical sample Parquet plus deterministic QC metadata. Only canonical
   chromosomes (1-22, X, Y in UCSC, plain, or RefSeq `NC_0000NN.V` spelling) are
   binned unless `--all-contigs` is set; mitochondrial, ALT, decoy, random,
   unplaced, and EBV contigs are skipped. `-t/--threads` is a CPU budget split
   between contig worker processes and per-handle htslib decode threads; output
   order follows the header.
2. **Reference:** Validate identical sample layouts, normalize depth, fit
   deterministic PCA, construct batched off-contig nearest-bin models, and
   write autosomal/optional sex branches to reference Parquet. With `--fasta`
   and no samples, build a flat uniform-depth model from contig lengths so
   predict can run without a control cohort (canonical chromosomes including
   X/Y are autosomal; `--all-contigs` includes every FASTA contig).
3. **Predict:** Project and normalize a sample, build within-sample references,
   segment with weighted Ruptures Binseg, call gains/losses, and write bin and
   segment BED (0-based half-open, extra columns with a header), calls BED,
   optional SVG, and matching Parquet tables. CLI `--prefix` receives the
   path prefix; canonical suffixes are appended by the tool.

A fixed human chromosome layout is intentionally absent, the tool is meant to be organism agnostic.

### CLI Structure

- `src/ccnv/cli.py`: The `ccnv` Typer entry point and subcommands.
- `src/ccnv/predict.py`: Functional logic for the predict stage.
- `src/ccnv/report.py`: BED, Parquet, and SVG writers for prediction artifacts.
- `src/ccnv/results.py`: Typed bin/segment/output objects shared by predict and report.
- `src/ccnv/convert.py`: Functional logic for the convert stage.
- `src/ccnv/reference.py`: Functional logic for creating reference files.
- `src/ccnv/segment.py`: Weighted L2 cost and Binseg segmentation.
- `src/ccnv/schemas.py`: Versioned Arrow schemas and metadata validation.
- `src/ccnv/models.py`: Typed configurations and genomic domain objects.

### Storage and coordinate contracts

- Schema version `1.0.0`; all coordinates are 0-based half-open.
- Sample rows are `contig`, `start`, `end`, and `count`.
- Reference rows add branch, mask, PCA, nearest-reference, and null-ratio model
  fields.
- Alignment-header contig names, order, lengths, and partial final bins are
  authoritative. Readers reject incompatible versions, layouts, and bounds.

## Development Workflow

### CI publish

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and hooks on
every push and pull request. Pull requests must change `CHANGELOG.md`. After
those checks pass:

- A push to `master` builds the Docker image and pushes it to GHCR tagged
  with the commit SHA and `latest`.
- A published GitHub Release builds a Python wheel, attaches it to that
  release, and pushes the image tagged with the commit SHA, the version
  (release tag with a leading `v` stripped), and `stable`.

Images live at `ghcr.io/<owner>/<repo>` (currently
`ghcr.io/matthdsm/copynumber-cnv`) as a multi-platform manifest for
`linux/amd64` and `linux/arm64`. The Dockerfile installs the `prod`
Pixi environment with `pixi install --locked` so the image matches
`pixi.lock`. The image `ENTRYPOINT` is `ccnv`.

### Pixi Tasks

Use `pixi run` for all standard development tasks to ensure environment consistency:

- `pixi run lint`: Check code quality with Ruff.
- `pixi run format`: Format code with Ruff.
- `pixi run test`: Run the test suite.
- `pixi run hooks`: Run pre-commit hooks (managed by `prek`).
- `pixi run docker-build`: Build the Docker image for `linux/amd64` and
  `linux/arm64` (Buildx cache; not loaded into the local daemon).

Run the threshold-free performance suite with:

```console
pixi run python benchmarks/run_benchmarks.py --scale 2 --repeats 3
```

### Code Style

- Adhere to the Ruff configuration in `pyproject.toml`.
- **Import Sorting:** Ruff handles this; ensure it's run before committing.
- **Typing:** Use maximal type annotations

### Code Changes
- All changes should be make through a pull request. Each pull request needs to be recorded in `CHANGELOG.md`, following best practices found [here](https://keepachangelog.com/en/1.1.0/#how)

### Testing Guidelines

- **Contract Tests:** When adding features that change how data is stored or processed, update the contract tests in `tests/`.
- **Functional Tests:** Aim for high coverage of the computational logic.
- **Pipeline Tests:** Keep BAM and CRAM end-to-end coverage in
  `tests/test_pipeline.py`.
- **Scientific Baselines:** Compare numeric agreement and call concordance
  against immutable `tests/fixtures/upstream/`; never require legacy byte
  compatibility.
- **Performance:** Record machine/input metadata in `docs/benchmarks.md`; do not
  add timing or RSS thresholds to CI.

## Agent Specific Instructions

- **Environment Awareness:** Always prefer `pixi run <command>` over direct execution if a pixi task exists.
- **Surgical Updates:** When fixing bugs, prioritize small, targeted changes and adding a corresponding test case.
- **Documentation:** Update this document and README if you make changes that affect the overall architecture or development workflow.

## Scientific limitations

Weighted-SSE Binseg penalties are not DNAcopy `alpha`. The compact synthetic and
published fixtures establish regression evidence, not clinical sensitivity,
specificity, or assay thresholds. Sex inference requires sufficient separated
cohort groups unless `--sex-cutoff` is set; per-sample sex labels are not
accepted on the CLI. See
`docs/validation.md` for accepted migration differences, `docs/benchmarks.md`
for measured evidence, and `docs/cli.md` for command flags.