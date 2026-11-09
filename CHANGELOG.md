# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Renamed the `convert` CLI subcommand and orchestrators to `extract` (`ExtractConfig`, `ExtractError`, `run_extract`), updating all module paths, test suites, test fixtures, and documentation with clean break semantics.
- Consolidated codebase into six cohesive domain subpackages (`ccnv.models`, `ccnv.stats`, `ccnv.io`, `ccnv.features`, `ccnv.qc`, `ccnv.report`) governed by a strict 7-layer unidirectional DAG, slimming root orchestrators (`extract.py`, `reference.py`, `ref_qc.py`, `predict.py`, `cli.py`) down to composition roots.
- Removed deprecated transitional root modules (`cohort.py`, `covariates.py`, `predict_qc.py`, `prenatal.py`, `prenatal_math.py`, `reference_compute.py`, `reference_qc.py`, `results.py`, `samplesheet.py`, `schemas.py`, `segment.py`, and `templates/`) and updated entire test suite imports to canonical subpackage paths.
- Enforced typed `Sequence[Cytoband] | None` contract for `ccnv.report.write_outputs` and `ccnv.report.html.render_genome_plot`, isolating file I/O strictly inside `ccnv.io.tracks`.
- Refactored codebase architecture to conform to thermonuclear code quality standards, eliminating code bloat, deduplicating logic, and ensuring all source and test modules remain strictly below 1,000 lines.
- Extracted modular test suites and test helpers into `tests/test_cohort.py`, `tests/test_reference_compute.py`, and `tests/helpers.py`, decomposing `tests/test_reference.py` from 1,065 to 763 lines.
- Replaced module-level mutable state in `src/ccnv/reference_compute.py` with immutable context dataclasses (`_NearestContext`, `_NullContext`).
- Streamlined `src/ccnv/reference.py` and `src/ccnv/prenatal.py` to ingest in-memory `Cohort` objects directly without redundant disk reads or table re-parsing.
- Optimized coordinate evidence slicing per-contig in `src/ccnv/predict.py` and hoisted invariant nearest-bin neighbor filters.
- Bumped Arrow Parquet storage schema to version `2.0.0` (`SCHEMA_VERSION = "2.0.0"`), introducing native multi-resolution reference branches (`prenatal_autosomal` at 100 kb, `prenatal_ffy` at 5 kb `chrY`) alongside 5 kb CNV branches (`autosomal`, `female_sex`, `male_sex`) and persisting scalar prenatal hyperparameters in `ccnv.prenatal` metadata.
- Replaced positional cohort sample arguments in `ccnv reference` with `--samplesheet / -s` for cohort references and `--fasta` for flat uniform-depth references.
- Added `--mode germline | prenatal` flag to `ccnv reference` and `ccnv predict` (defaulting to `germline`) with strict reference mode validation.
- Updated `ccnv extract` sample metadata to track effective read counts examined, kept, and filtered strictly on target canonical contigs (`effective_reads_examined`, `effective_reads_kept`, `autosomal_reads_kept`, `sex_reads_kept`).
- Extended prediction BED (`_segments.bed.gz`, `_calls.bed.gz`) and VCFv4.2 (`_segments.vcf.gz`, `_calls.vcf.gz`) outputs in `--mode prenatal` to include expected copy ratios (`EXP_LOG2R`), implied fetal fractions (`IMPLIED_FF`), and discordance annotations (`DISCORDANT`, `DISC_REASON`).
- `ccnv ref-qc` and `ccnv reference` now evaluate and record prenatal quality metrics in `<prefix>_qc.json` when auditing or building prenatal references.
- Replaced the static `_plot.svg` output in `ccnv predict` with a minimal, interactive Plotly HTML viewer (`_plot.html`) rendered from an external Jinja2 template (`src/ccnv/templates/genome_plot.html.j2`), featuring whole-genome pan/zoom, coordinate and cytoband search, configurable Z-score and $\log_2$ ratio threshold color-coding, base64-encoded binary typed arrays (`scattergl`), and an optional `--cytobands` ideogram track.
- Rewrote `README.md`, `docs/cli.md`, `docs/pipeline.md`, and `AGENTS.md` to be clear and concise, crediting the published statistical methodology of WISECONDOR (Straver et al., 2014) and WisecondorX (Raman et al., 2019).
- Prediction BED outputs (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`) are now always bgzip-compressed and tabix-indexed (`.tbi`), with `#contig` headers for standard tabix/IGV compatibility.
- Prediction outputs (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`, `_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`, and `_plot.html`) now sort chromosomes and intervals in natural sort order (e.g. `chr1`, `chr2`, ..., `chr10`, ..., `chrX`, `chrY`) instead of lexicographical order.
- Parallelized `ccnv reference` execution controlled by a `-t/--threads` flag (defaulting to system CPU count), multi-processing cohort sample loading and rebinning via `src/ccnv/cohort.py`, parallelizing batched off-contig nearest-bin selection via `src/ccnv/reference_compute.py`, and multi-processing null model derivation.
- Extracted cohort ingestion and validation into `src/ccnv/cohort.py` and compute kernels into `src/ccnv/reference_compute.py`, keeping modules modular and strictly below 1,000 lines.
- Replaced boolean array masking with precomputed contiguous contig slice bounds in `select_nearest_bins`, accelerating per-query candidate masking by over 400x.
- Vectorized `rebin_sample_table` contig boundaries using NumPy run-length transitions instead of converting Arrow contig columns to Python string lists.
- Optimized reference schema (`SCHEMA_VERSION = "1.0.0"`), replacing variable-length `null_ratios` and `nearest_ref_distances` with pre-filtered sorted `nearest_ref_indices`, scalar `mean_distance`, `null_mean`, and `null_std`.
- Reference Parquet writing now uses Zstandard (level 3) compression, dictionary encoding on categorical strings (`branch`, `contig`), and Delta Binary Packing on coordinates (`start`, `end`) and neighbor indices (`nearest_ref_indices.list.element`), shrinking reference files by ~6.5x (e.g., from 2.16 GB down to 333 MB).
- Vectorized `predict.py`, `convert.py`, `reference_qc.py`, and `schemas.py` using PyArrow chunk buffer slicing and NumPy vectorized ops, slashing reference reading, bounds checking, and normalization time by over 5x.
- Unified default bin size to 5,000 bp across `convert`, `reference`, and models.
- `ccnv reference` accepts FASTA inputs as a positional argument alongside sample Parquet inputs, removing the `--fasta` option.
- `ccnv reference` rebins sample counts to `--bin-size` when cohort samples differ in native bin size, requiring matching contig lengths.
- `ccnv predict` rebins sample counts in memory to match the reference bin coordinates when bin sizes differ, without modifying the reference file.
- Convert and FASTA reference now share one contig-layout selector (`select_layout`) so canonical filtering is no longer duplicated, and `GenomeProfile` is the in-memory layout type. Prediction I/O lives in `ccnv.report`, with result types in `ccnv.results`. Bin coordinates are read through `schemas.bin_coordinates`.
- CLI tests compare Rich output after stripping ANSI so CI color does not split flags such as ``--loglevel``.

### Removed

- Dropped Parquet outputs (`_bins.ccnv`, `_segments.ccnv`) from `ccnv predict` in favor of standard bgzipped/tabix-indexed BED and VCF deliverables.

### Fixed

- Resolved `UnboundLocalError` for `fasta` variable in `ccnv reference` CLI command.
- Scaled depth normalization strictly over unmasked bins in reference build and prediction coverage, and guarded PCA reconstruction against underdetermined active bin rank.

### Added

- Cell-free DNA (cfDNA) prenatal analysis mode (`--mode prenatal`): integrated prenatal screening into `ccnv reference` and `ccnv predict`, estimating fetal fraction via `FFY`, `FFX`, and `PREFACE`, classifying fetal sex and sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`), and evaluating CNV segment discordance against active fetal fraction expectations.
- NumPy, SciPy, and scikit-learn prenatal math kernels (`src/ccnv/prenatal_math.py`): implemented 1D 2-component Gaussian mixture model (`infer_sex_gmm_1d` via `sklearn.mixture.GaussianMixture` and `scipy.optimize.brentq`), non-negative Lasso regression (`fit_lasso_fista` via `sklearn.linear_model.Lasso` and `scipy.optimize.nnls`), centered unscaled truncated PCA (`pca_irlba_numpy` via `sklearn.impute.SimpleImputer`, `sklearn.decomposition.PCA`, and `sklearn.utils.extmath.svd_flip`), L-BFGS-B Sigmoid neural network (`fit_rprop_mlp` via `scipy.optimize.minimize`), Ordinary Linear Model (`train_olm` via `sklearn.linear_model.LinearRegression`), OLS skew correction (`ols_fit` via `scipy.stats.linregress`), and robust Huber regression (`rlm_huber` via `sklearn.linear_model.HuberRegressor`), replacing PyTorch and ONNX Runtime dependencies.
- Prenatal reference and prediction orchestration (`src/ccnv/prenatal.py`): in-memory multi-resolution coarsening from 5 kb to 100 kb (`--prenatal-binsize`, default 100000 bp), full prenatal reference fitting (`fit_prenatal_models`), single-sample prenatal inference (`predict_prenatal`), active fetal fraction selection, and segment discordance annotation (`annotate_prenatal_segments`).
- Cohort samplesheet ingestion (`src/ccnv/samplesheet.py`): added `-s/--samplesheet` to `ccnv reference` accepting root-list YAML/JSON cohort manifests validated against `src/ccnv/samplesheet.schema.json`, with path resolution and optional `sex` and `ff` overrides.
- Prediction quality control reporting (`src/ccnv/predict_qc.py`): `ccnv predict` generates `<prefix>_qc.json` on every run, capturing effective read counts, filter drops on target contigs, bin summaries, and structured prenatal results.
- Case B prenatal segment scaling and discordance annotation (`src/ccnv/prenatal.py`): computes expected ratio $\log_2(1 \pm 0.5 \cdot \text{active\_ff})$ and implied fetal fraction, flagging sub-fetal noise (`SUB_FETAL_RATIO`), super-fetal maternal alterations (`SUPER_FETAL_RATIO`), and gonosome changes (`SCA_DISCORDANT`).
- High-contrast prenatal visualization (`src/ccnv/templates/genome_plot.html.j2`): renders discordant segments with high-visibility purple strokes (`width: 6.0`) and diamond markers, displays horizontal dashed reference lines at $\log_2(1 \pm \text{active\_ff}/2)$, and presents a prenatal summary header badge with fetal sex, active FF, estimator breakdown, and SCA call.
- Diploid `XX` gonosome normalization baseline: in prenatal prediction, `chrX` bins are normalized against the diploid `XX` (`female_sex`) reference branch to quantify fetal sex-chromosome dosage shifts relative to maternal baseline.
- Covariate-aware read normalization (`src/ccnv/covariates.py`): added `--mappability` (BigWig `.bw`/`.bigWig`), `--gc` (BigWig, BED/bedGraph/TSV, or indexed FASTA), and `--min-mappability` (default `0.5`) flags to `ccnv reference` and `ccnv predict`. Reference `.ccnv` files now persist per-bin `gc` and `mappability` (`float32`) columns, automatically derive GC fraction and `>50%` `N`-base masking from the input FASTA in flat reference mode, select off-contig nearest bins in flat mode using standardized `(gc, mappability)` feature distance, and apply outlier-resistant Huber IRLS polynomial trend removal per sample before PCA in both cohort and flat workflows.
- VCFv4.2 prediction outputs: `ccnv predict` now generates bgzip-compressed and tabix-indexed VCF files for bins (`_bins.vcf.gz`), segments (`_segments.vcf.gz`), and called events (`_calls.vcf.gz`) using native `pysam.VariantFile` and `pysam.BGZFile` single-pass streaming, containing standard symbolic SV tags (`<DUP>`, `<DEL>`, `<CNV>`), 1-based intervals, and copy ratio / z-score format metrics, enabled by default with `--vcf / --no-vcf` control.
- `ccnv ref-qc`: evaluate reference model usability and quality heuristics (`PASS`, `WARN`, `FAIL`) on existing reference files, emitting `<prefix>_qc.json` and a stderr summary.
- Reference creation quality control: `ccnv reference` automatically evaluates reference metrics and writes `<prefix>_qc.json` during build (toggleable via `--qc-report / --no-qc-report`).
- Comprehensive pipeline architecture documentation (`docs/pipeline.md`) with detailed Mermaid flowcharts for all pipeline subcommands (`convert`, `reference`, `ref-qc`, `predict`), and embedded stage flowcharts in the CLI reference (`docs/cli.md`).
- `rebin_counts` and `rebin_sample_table` for 1D mass-conserving count interpolation across bin grids.
- `ccnv convert`: stream BAM/CRAM into versioned sample Parquet, auto-index missing `.bai`/`.crai`, bin canonical chromosomes (or `--all-contigs`), and split `--threads` across contig workers and htslib decode.
- `ccnv reference`: build autosomal and optional sex-contig models from control Parquet, or a flat uniform-depth model from FASTA when there is no cohort.
- `ccnv predict`: PCA-normalize, segment with weighted Binseg, call gains and losses, and write bgzipped/indexed BED, VCFv4.2, and optional SVG.
- Versioned Parquet contracts (`1.0.0`), header-authoritative contig layouts, and organism-agnostic naming (no fixed 24-chromosome genome).
- Sex-contig branches inferred from a Y-signal contig (`--sex-cutoff` optional); no per-sample sex labels on the CLI.
- CI that publishes multi-arch (`linux/amd64`, `linux/arm64`) GHCR images from `pixi.lock` (`latest` on `master`; SHA, version, and `stable` on release) and attaches a Python wheel to GitHub Releases. Local images: `pixi run docker-build`.
- Real-world NIPT conversion, reference creation, and prediction benchmarks documented in `docs/benchmark.md`.
- CI rejects pull requests that do not update `CHANGELOG.md`.
