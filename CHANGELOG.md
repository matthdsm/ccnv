# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

## [Unreleased]

### Added

- Zensical documentation site hosted on GitHub Pages (`https://matthdsm.github.io/copynumber-cnv/`), with automated deployment in `.github/workflows/docs.yml`, MathJax 3 LaTeX math support, and pixi tasks (`pixi run docs-build`, `pixi run docs-serve`).
- Codified documentation policy in `AGENTS.md` requiring all documentation amendments to be integrated into `docs/` and tested against `zensical build --clean --strict`.
- Core pipeline subcommands: `ccnv extract` (stream BAM/CRAM into sample Parquet with deterministic read QC and canonical contig filtering), `ccnv reference` (build cohort or flat uniform-depth FASTA models with Gavish-Donoho SVHT PCA and signal-subspace `KDTree` neighbor search), `ccnv ref-qc` (audit reference usability with `PASS`/`WARN`/`FAIL` heuristics), and `ccnv predict` (robust Huber PCA projection, weighted `Binseg`/`KernelCPD` segmentation, autocorrelation-corrected Stouffer Z-score calling).
- Cell-free DNA (cfDNA) prenatal analysis mode (`--mode prenatal`): non-negative Elastic Net 5 kb `chrY` `FFY`, convex logit-link `RidgeCV`/`ElasticNetCV` 100 kb `FFA`, Huber `FFX`, 2D stoichiometric Gaussian Bayes sex chromosome aneuploidy classifier with Mahalanobis $\chi^2$ outlier gate, precision-fused `active_ff`, and $\mathrm{LLR}_{\text{fetal}}$ segment discordance scoring (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`).
- Covariate normalization and robust depth scaling: joint natural cubic B-spline regression with interaction terms (`SplineTransformer` + `HuberRegressor`) for GC content and mappability bias correction (`--gc`, `--mappability`, `--min-mappability`), Smyth Empirical Bayes moderated variance shrinkage (`squeeze_var`), and DESeq2 median-of-ratios library size factor normalization.
- Deliverables and interactive reporting: bgzipped and tabix-indexed BED (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`) and standard VCFv4.2 outputs (`_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`), structured QC payloads (`<prefix>_qc.json`), and standalone interactive Plotly WebGL HTML viewer (`_plot.html`) with pan/zoom, cytoband search, and prenatal discordance overlays.
- Native performance and storage architecture: versioned Parquet storage schema (`2.0.0`) with Zstandard compression and Delta Binary Packing, Polars DataFrame expressions with zero-copy Arrow interop, mass-conserving 1D count rebinning, multi-threaded C-compiled `scipy.spatial.KDTree` neighbor search, and organism-agnostic contig handling.
- Documentation, benchmarks, and CI automation: mathematical methodology specification with 51 citations (`docs/methodology.md`), WisecondorX algorithmic comparison and runtime benchmarks (`docs/benchmark.md`), pipeline flowcharts (`docs/pipeline.md`), and GitHub Actions workflow building multi-arch container images (`ghcr.io/matthdsm/copynumber-cnv`).
