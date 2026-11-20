# `ccnv` Agent Guidelines

This document provides project-specific instructions and architectural context for agents working on the `ccnv` codebase.

## Project Overview

`ccnv` is an independent Python bioinformatics toolkit for Copy Number Variation (CNV) analysis in whole-genome sequencing (WGS) data at any coverage, building on published within-sample reference normalization methodology (Straver et al., 2014; Raman et al., 2019).

### Core Stack

- **Language:** Python 3.14+
- **Compiled Extensions:** Rust PyO3 (`abi3-py314`) extension (`crates/ccnv-core`, `ccnv._core`) wrapping `noodles` (BAM/CRAM/FASTA/BGZF/Tabix/CSI) and `bigtools` (BigWig)
- **CLI Framework:** Typer
- **Data Science:** NumPy, SciPy, Polars, Scikit-learn, PyArrow, Ruptures
- **Serialization & Validation:** PyYAML, jsonschema
- **Templating:** Jinja2
- **Environment & Dependency Management:** [Pixi](https://pixi.sh). The `default` environment includes the `dev` feature; `prod` contains the runtime dependencies without test/lint tools and is installed in the Docker image; `docs` is an isolated documentation environment containing Zensical.
- **Linting:** Prek & Ruff
- **Testing:** Pytest (using `unittest` style tests)

## Architectural Context

### Pipeline Stages

1. **Extract:** Stream a BAM/CRAM once per contig via compiled Rust `ccnv._core` scanner (`noodles-bam` and `noodles-cram`), filter reads, bin paired-end proper pairs by dyad midpoint (`TLEN > 0`) or single-end reads by 0-based `reference_start`, and write canonical sample Parquet plus deterministic QC metadata. Populates per-bin `count`, `paired_count`, `frag_len_sum`, `short_frag_count` ($`[80, 150]\text{ bp}`$), `endpoint_count`, `frag_gc_sum`, and 24 fixed-edge piecewise-uniform (variable-width: 20 bp, 10 bp, 50 bp, and 201 bp) fragment length bands `frag_bands`, and writes `ccnv.fragmentomics` metadata containing sample-level size landmarks, 571-bin length histograms, and 256-element 5' 4-mer motif counts. Only canonical chromosomes (1-22, X, Y in UCSC or plain spelling) are binned unless `--all-contigs` is set; mitochondrial, ALT, decoy, random, unplaced, and EBV contigs are skipped. `-t/--threads` is a CPU budget for parallel tile streaming and decompression; single-threaded extraction streams sequentially without requiring an index, while multi-threaded tile extraction partitions contigs into 50 Mb tiles using `.bai`/`.csi`/`.crai` indexes. Output order follows the header. Sample QC metadata tracks effective reads examined, kept, and filtered strictly across binned target contigs.
2. **Reference:** Ingest cohort samples via a root-list YAML/JSON samplesheet (`-s/--samplesheet`) or build a flat uniform-depth model from a reference FASTA (`--fasta`). Validate sample layouts (rejecting mixed paired-end/single-end cohorts), rebin sample buffers to the target bin size (default 5000 bp) when contig lengths match (preserving channels via a vectorized NumPy `reduceat` fast-path when coarsening integer-multiple grids, or redistributing `count` only via cumulative knot interpolation while zeroing fragmentomic channels and dropping `ccnv.fragmentomics` metadata on non-nested grids), normalize depth using DESeq2 median-of-ratios size factors over 1 Mb super-blocks, optionally load GC content (`--gc`) and mappability BigWig (`--mappability`) tracks via `noodles`/`bigtools` to mask low-mappability/high-N bins (`--min-mappability`, default 0.5) and remove per-sample multiplicative GC, mappability, and fragmentomic bias via zero-inclusive `PoissonRegressor` regression (using 200 bp GC quantiles on coarse grids $`\ge 25{,}000\text{ bp}`$ or quantile B-splines otherwise), fit deterministic PCA with Gavish-Donoho SVHT rank selection over variance-whitened residuals, construct global signal-space Prototype Graph models ($`M \le 2{,}048`$) querying $`k=4`$ nearest prototypes with soft inverse-distance weights and leave-one-contig-out (LOCO) donor evaluation, evaluate Smyth Empirical Bayes moderated biological overdispersion (`squeeze_var`, `bio_dispersion`), and write reference Parquet with reference QC (`<prefix>_qc.json`). Supports `--mode germline | prenatal` (default `germline`). In prenatal mode (`--mode prenatal`), fits 5 kb `chrY` `FFY` (via closed-form Poisson MSY MLE), coarsens 5 kb counts in memory to 100 kb (`--prenatal-binsize`, default 100000 bp) for `FFA` (unscaled PCA + convex logit-link `RidgeCV`/`ElasticNetCV` ensemble) and forward-calibrated $`R_X`$ baseline regression, deconvolves fetal and maternal fragment length densities ($`\hat{p}_F(L)`$ and $`\hat{p}_M(L)`$), applies reference-side 24-band Fisher read weighting $`w^{\ast}_{0.08}`$, fits a 36-feature standardized logit-link `FF_size` ensemble, partitions `female_sex` (diploid `XX` baseline) and `male_sex` branches using prenatal sex classification, and persists native `prenatal_autosomal` (100 kb) and `prenatal_ffy` (5 kb) branches along with `ccnv.prenatal` scalar metadata.
3. **Ref-QC:** Audit an existing reference Parquet file and emit `<prefix>_qc.json` and a stderr usability verdict (`PASS`, `WARN`, `FAIL`), including a prenatal summary block for prenatal references.
4. **Predict:** Project and normalize a sample loaded from `.bam`, `.cram` (with `--fasta`), or `.ccnv` Parquet (streaming directly from alignments or rebinning sample counts in memory to reference bin coordinates), scale depth via 1 Mb super-blocks, apply stored or overridden (`--gc`, `--mappability`, `--min-mappability`) per-sample GC, mappability, and fragmentomic normalization before robust `HuberRegressor` PCA projection, build within-sample references via weighted prototype medians, synthesize depth-decomposed Empirical Bayes inverse-variance precision weights, pre-split contigs at masked gaps $`\gt 2\text{ Mb}`$ into contiguous euchromatic arm blocks, segment with calibrated `ruptures.Pelt` using mBIC penalty (using C-compiled `KernelCPD` when weights are uniform), score segments via dyadic multi-scale BLUE Z-scores, call gain/loss copy-number events via discrete integer copy-number Bayes factors ($`CN \in \lbrace 0, 1, 2, 3, 4+\rbrace`$) resolving homozygous deletions ($`CN=0`$) down to $`0.1\times`$ sequencing coverage, and write single-pass bgzipped/indexed bin, segment, and call BED, optional VCFv4.2 outputs (`--vcf`, with opt-in `--bins-vcf`), interactive Plotly HTML (`_plot.html`, with bundled GRCh38 cytobands auto-loaded when canonical contig lengths match `hg38` or overridden via `--cytobands`), and sample prediction QC (`<prefix>_qc.json`). Supports `--mode germline | prenatal` (default `germline`) with strict reference mode validation and optional `--twins` for known twin gestations. In prenatal mode, normalizes `chrX` bins against the diploid `XX` (`female_sex`) reference branch, evaluates `FFY`, `FFX`, `FFA` (with degenerate $`\lVert \mathbf{w}_{\text{olm}} \rVert_2`$ variance scaling via `effective_ffa_sigma`), and `FF_size`, classifies sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`) via a 4D erfc ray-marginalized Gaussian mixture classifier alongside 2-fetus twin stoichiometry (`FF`, `FM`, `MM`, `twin_probability`, `male_ff`, `female_ff`) via `classify_twin_pregnancy` (preserving both `sca` and `twin` in `<prefix>_qc.json`), selects Student-$`t_4`$ IRLS precision-fused `active_ff`, normalizes male-fetus (`fetal_sex == "M"`) 5 kb clean MSY `chrY` bins from `prenatal_ffy` against an FF-calibrated 1-copy Poisson baseline via `normalize_prenatal_chry` (scaling to `male_ff` in `FM` twins) for haploid ($`CN_0 = 1`$) segmentation, BED/VCF output, and HTML plotting (omitting `chrY` when `fetal_sex == "F"`), annotates segments with expected ratios, implied fetal fractions, discordance status/reasons (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`), short fragment ratio (`frag_short_ratio`), fragment length LLR (`frag_llr_size`), and biological origin call (`origin_call`) via a 5-state self-referenced coupled depth + Dirichlet-Multinomial origin engine (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$, calibrated to single-fetus and concordant-twin fetal fractions when `gestation == "twin"`), highlights discordant segments in high contrast within the HTML viewer, and records prenatal and fragmentomic metrics and warnings in `<prefix>_qc.json`.

A fixed human chromosome layout is intentionally absent; the tool is organism-agnostic.

### Module Structure

- `src/ccnv/cli.py`: The `ccnv` Typer entry point and subcommands (`extract`, `reference`, `ref-qc`, `predict`).
- `src/ccnv/extract.py`: Root orchestrator for the extract stage, parallel tile streaming via `ccnv._core`, and effective read statistics collection.
- `src/ccnv/reference.py`: Root orchestrator for creating cohort and flat FASTA reference files across germline and prenatal modes.
- `src/ccnv/ref_qc.py`: Root orchestrator for auditing reference Parquet files and emitting usability verdicts.
- `src/ccnv/predict.py`: Root orchestrator for single-sample prediction, within-sample prototype normalization, calibrated Pelt segmentation, BLUE scoring, Bayes factor calling, and reporting.
- `src/ccnv/models/`: Pure domain types, dataclasses, sample/branch array buffers, schemas, configuration containers, metadata serializers, and exceptions (`domain`, `buffers`, `config`, `results`, `schemas`, `metadata`, `validation`, `exceptions`).
- `src/ccnv/io/`: Pure I/O ingestion and persistence routines for alignments, external tracks (GC, mappability, cytobands, including bundled GRCh38 `cytoBand_hg38.gz`), samplesheet schemas, and versioned Parquet tables (`alignments`, `tracks`, `samplesheet`, `parquet`).
- `src/ccnv/stats/`: Pure stateless mathematical and machine learning kernels for Gavish-Donoho SVHT PCA, Prototype Graph search, Smyth Empirical Bayes variance shrinkage, 1 Mb super-block scaling, weighted segmentation costs and dyadic BLUE scoring, Poisson and B-spline regression, prenatal mathematical estimators, and fragmentomics algorithms (`pca`, `neighbors`, `segmentation`, `superblocks`, `regression`, `prenatal_math`, `fragmentomics`).
- `src/ccnv/features/`: In-memory feature transforms, binning, cohort alignment/rebinning, 1 Mb super-block DESeq2 depth scaling, Poisson covariate bias correction, prenatal reference/inference workflows, sex inference, and fragmentomics feature extraction (`binning`, `rebinning`, `cohort`, `covariates`, `normalization`, `sex`, `prenatal`, `prenatal_grid`, `prenatal_origin`, `fragmentomics`).
- `src/ccnv/qc/`: Quality control metrics calculation, verification, and JSON serialization for samples, references, and predictions (`sample`, `reference`, `predict`).
- `src/ccnv/report/`: Genomic deliverable generators for bgzipped/tabix BED, VCFv4.2, and interactive Plotly HTML viewer (`bed`, `vcf`, `html`, `templates/genome_plot.html.j2`).
- `crates/ccnv-core/`: High-performance Rust extension (`ccnv._core`) wrapping `noodles` and `bigtools` for BAM/CRAM streaming, FASTA GC quantile scanning, BigWig overlap aggregation, and single-pass BGZF/tabix/CSI writing.

### Storage and Coordinate Contracts

- Schema version `2.0.0`; all internal and BED coordinates are 0-based half-open.
- Sample rows store 7 coordinate-free per-bin columns (`count`, `paired_count`, `frag_len_sum`, `short_frag_count`, `endpoint_count`, `frag_gc_sum`, and `frag_bands`), with bin coordinates reconstructed from `ccnv.contig_order`, `ccnv.contig_lengths`, and `ccnv.bin_size`; metadata also includes `ccnv.qc` effective read metrics on target contigs and `ccnv.fragmentomics` (dropped only after non-nested in-memory rebinning).
- Reference rows store 17 columns (`branch`, `contig`, `start`, `end`, `masked`, `gc`, `mappability`, `gc_quantiles`, `ref_frag_len`, `ref_ep_rate`, `pca_mean`, `pca_loadings`, `proto_ids`, `proto_weights`, `mean_distance`, `null_mean`, and `bio_dispersion`). Branches include CNV models (`autosomal`, `female_sex`, `male_sex`) and prenatal models (`prenatal_autosomal` at 100 kb, `prenatal_ffy` at 5 kb `chrY`), with scalar prenatal hyperparameters stored under `ccnv.prenatal` metadata.
- Alignment-header contig names, order, lengths, and partial final bins are authoritative. Readers reject incompatible versions, layouts, and bounds.

## Development Workflow

### CI Publish

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and hooks on every push and pull request. Pull requests must change `CHANGELOG.md`. After those checks pass:

- A push to `master` builds the Docker image and pushes it to GHCR tagged with the commit SHA and `latest`.
- A published GitHub Release builds multi-platform Python `abi3` wheels (`ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest`) and `sdist`, attaches them to that release, and pushes the image tagged with the commit SHA, the version (release tag with a leading `v` stripped), and `stable`.

Images live at `ghcr.io/<owner>/<repo>` (currently `ghcr.io/matthdsm/ccnv`) as a multi-platform manifest for `linux/amd64` and `linux/arm64`. The Dockerfile compiles the Rust extension inside the `build` stage and installs the `prod` Pixi environment with `pixi install --locked` (stripping compiler artifacts like `rustlib`), so the runtime image is slim and free of `rustc`, `pysam`, or `pybigwig`. The image `ENTRYPOINT` is `/app/entrypoint.sh` (which activates the `prod` Pixi environment) with default `CMD ["ccnv"]`.

### Pixi Tasks

Use `pixi run` for all standard development tasks to ensure environment consistency:

- `pixi run lint`: Check code quality with Ruff.
- `pixi run format`: Format code with Ruff.
- `pixi run test`: Run the test suite.
- `pixi run hooks`: Run pre-commit hooks (managed by `prek`).
- `pixi run docker-build`: Build the local `linux/amd64` Docker image (`ghcr.io/matthdsm/ccnv:dev`).
- `pixi run docs-build`: Build the documentation site in the `docs` environment with Zensical (`pixi run -e docs zensical build --clean --strict`).
- `pixi run docs-serve`: Serve the documentation site locally in the `docs` environment with Zensical (`pixi run -e docs zensical serve`).
- `pixi run bump-version <version> [--date YYYY-MM-DD] [--allow-empty]`: Bump application version across `pyproject.toml`, `src/ccnv/__init__.py`, `Cargo.toml`, `Cargo.lock`, `galaxy/macros.xml`, pre-1.0 alpha banners (`README.md`, `CHANGELOG.md`, `docs/index.md`), and finalize `CHANGELOG.md`.
- `pixi run bump-schema <version|major|minor|patch> [--keep-previous]`: Bump Parquet `SCHEMA_VERSION` and `SUPPORTED_SCHEMA_VERSIONS` across `src/ccnv/models/metadata.py`, docstrings, Markdown documentation, and contract tests.

### Code Style

- Adhere to the Ruff configuration in `pyproject.toml`.
- **Import Sorting:** Ruff handles this; ensure it is run before committing.
- **Typing:** Use maximal type annotations.

### Code Changes and Versioning

- **Code Changes:** All changes should be made through a pull request. Each pull request needs to be recorded in `CHANGELOG.md`, following best practices found [here](https://keepachangelog.com/en/1.1.0/#how).
- **API Reference Synchronization:** Any relevant code change under `src/ccnv/` or `crates/ccnv-core/`—adding, removing, moving, or renaming modules, classes, public or internal (`_`) functions, methods, properties, or `ccnv._core` bindings, altering parameter signatures or return/behavioral contracts, or modifying cross-module call-graph edges—must be reflected in `docs/api.md` (both the module symbol reference and the Section 1 Mermaid call-graph DAGs) in the same pull request.
- **Versioning:** Follow semantic versioning for releases, using `pixi run bump-version <version>` to synchronize `pyproject.toml`, `src/ccnv/__init__.py`, `Cargo.toml`, `Cargo.lock`, `galaxy/macros.xml`, and `CHANGELOG.md`, and `pixi run bump-schema <version|major|minor|patch>` when evolving the Parquet storage schema.

### Testing Guidelines

- **Contract Tests:** When adding features that change how data is stored or processed, or when modifying modules, callables, signatures, or call graphs, update the contract tests in `tests/` (`tests/test_contracts.py` and `tests/test_docs_markdown.py`, which enforces bidirectional AST module, symbol, parameter-signature, and Mermaid DAG parity with `docs/api.md`).
- **Functional Tests:** Aim for high coverage of the computational logic.
- **Pipeline Tests:** Keep BAM and CRAM end-to-end coverage in `tests/test_pipeline.py`.

## Agent Specific Instructions

- **Environment Awareness:** Always prefer `pixi run <command>` over direct execution if a pixi task exists.
- **GitHub Actions:** Ensure github actions are using the latest version at all times, updating the workflow files as needed. Reverting to older versions is strictly forbidden.
- **Surgical Updates:** When fixing bugs, prioritize small, targeted changes and adding a corresponding test case.
- **Documentation:** Update this document, `README.md`, `docs/api.md`, and the documentation site in `docs/` whenever you make changes that affect modules, functions, classes, signatures, call graphs, architecture, CLI, or development workflow. All documentation amendments must be integrated into the site (`docs/`), registered in `zensical.toml` navigation where appropriate, and verified for strict Zensical build compatibility (`pixi run docs-build` or `pixi run -e docs zensical build --clean --strict`) and Markdown/API contract compliance (`tests/test_docs_markdown.py`).
- **Markdown Formatting & Dual-Target Rendering Rules:** Markdown files in this repository target two distinct renderers enforced by `tests/test_docs_markdown.py`:
  - **Shared Rules (All `.md` Files):**
    - Do not hard-wrap lines in Markdown files; rely on editor soft-wrapping for line length. Keep paragraphs and list items on a single logical line.
    - Avoid unnecessary newlines, redundant blank lines, and empty lines inside Mermaid diagram blocks. Quote node labels containing special characters (parentheses, brackets).
    - Escape all pipe characters inside Markdown tables, including within LaTeX math expressions (use `\|` or `\vert` instead of bare `|`).
    - Ensure LaTeX math expressions use standard MathJax/KaTeX syntax, avoid leading or trailing whitespace inside delimiters (`$x$` rather than `$ x $`), wrap compound superscripts in braces (`^{\ast}`, `^{\top}`), and use `\mathrm{...}` or `\text{...}` for multi-letter operator names (`\operatorname` is disallowed).
    - Never place percent signs `%` inside LaTeX math delimiters (`$...$` or `$$...$$`); `%` acts as a TeX comment and swallows closing delimiters. Keep `%` outside math spans (e.g. `50%` or ``$`\le 5`$%``).
    - Use `\lt` and `\gt` instead of bare `<` and `>` inside LaTeX math to prevent HTML tag parsing collisions.
    - Always place display math `$$` delimiters on their own separate lines at column 0 (never inline `$$expr$$` and never indented inside lists) and separate block-level elements (headings, tables, blockquotes, admonitions, code blocks, lists, and `$$` blocks) from surrounding prose by blank lines.
    - Use valid repository-relative links for internal files and anchors; never include absolute local filesystem paths, `file:///` URLs, or unlinked `` `[src/ccnv/...]` `` pseudo-links.
  - **Zensical Site Markdown (`docs/**/*.md`):**
    - Must render cleanly under Zensical (`pymdownx.arithmatex` + MathJax v3 + `admonition`).
    - Use standard `$...$` (or `\(...\)`) for inline math. Never use GitHub's backtick-dollar syntax (`` $`...`$ ``) inside `docs/`, because Python-Markdown parses backticks into `<code>` tags before `arithmatex` runs and corrupts surrounding prose.
    - Inside `$$...$$` `\begin{cases}` environments, use standard single row breaks `\\` (never `\\\\`, which renders empty extra rows in Zensical).
    - Use Zensical admonition blocks (`!!! note`, `!!! warning`, `!!! tip`, `!!! important`) instead of GitHub blockquote alerts (`> [!NOTE]`).
  - **GitHub Flavored Markdown (`README.md`, `AGENTS.md`, `CHANGELOG.md`, `LICENSE.md`, `src/**/README.md`, `tests/**/README.md`):**
    - Must render cleanly on GitHub (`cmark-gfm` + MathJax).
    - Use GitHub's backtick-dollar syntax (`` $`...`$ ``) whenever inline math is adjacent to hyphens, en-dashes, or parentheses (e.g. ``inverse-$`\chi^2`$``, ``$`(P+2)`$-dimensional``) or contains braces, underscores, or asterisks (`{`, `}`, `_`, `*`) that could trigger CommonMark emphasis (`<em>`).
    - Inside `$$...$$` display math, use `\lbrace`/`\rbrace` instead of `\\{`/`\\}`, `^{\ast}` instead of `^{*}`, `\quad` instead of `\;`, and `\\\\` for row breaks inside `\begin{cases}` so CommonMark backslash-unescaping does not strip TeX tokens.
    - Use GitHub blockquote alerts (`> [!NOTE]`, `> [!WARNING]`, etc.) instead of Zensical `!!!` admonitions.
