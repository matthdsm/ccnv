# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

## [Unreleased]

### Added

- Cell-free DNA (cffDNA vs maternal cfDNA) fragmentomic feature extraction and size distribution modeling across `extract`, `reference`, and `predict`:
  - `ccnv extract`: Single-pass paired-end alignment streaming with dyad midpoint binning, extracting insert length ($30$–$600\text{ bp}$), unclipped $5'$ 4-mer motif codes, fragment GC content, and endpoint counts, serializing cohort fragmentomics under `ccnv.fragmentomics` Parquet metadata.
  - Sizing shift cross-correlation alignment ($\Delta_{\text{bp}} \in [-15, 15]$) standardizing platform- and run-specific sizing calibration offsets across cohorts.
  - Multi-modal regularized `FF_size` linear ensemble incorporating mode-aligned length bands, $10.4\text{ bp}$ helical Fourier cleavage power, and 256-dimensional $5'$ 4-mer motifs (Motif Diversity Score MDS).
  - Cohort fragment length deconvolution estimating pure fetal $\hat{p}_F(L)$ and maternal $\hat{p}_M(L)$ probability densities from female controls and high-confidence male samples.
  - Depth-conserving Fisher-optimal read weighting $w^{\ast}(L)$ evaluated via bracketed root-finding with $\mathbb{E}_{L \sim p_M}[w^{\ast}(L)] = 1.0$, boosting prenatal CNV signal while strictly preserving diploid coverage.
  - Multi-channel 4-feature rebinning (`count`, `fragment_lengths`, `endpoint_count`, `frag_gc_sum`) supporting arbitrary target bin grids.
  - Augmented Huber B-spline regression and $\mathbb{R}^{P+4}$ KDTree off-contig neighbor matching incorporating mean fragment length $\bar{L}_i$ and endpoint density ratio $E_i / \max(C_i, 1.0)$.
  - Multi-modal precision fusion combining `FFY`, `FFA`, `FFX`, and `FF_size` into optimal Bates-Granger `active_ff`.
  - Segment-level fragmentomic log-likelihood ratio scoring ($\mathrm{LLR}_{\text{size}}$) and biological origin classification (`FETAL_CONFIRMED`, `MATERNAL_CONFIRMED`, `INDETERMINATE`).
  - Polymorphic alignment ingestion in `ccnv predict`: directly accepts `.bam`, `.cram` (with `--fasta`), and `.parquet`/`.ccnv` sample files.
  - Deliverables: annotated `frag_short_ratio`, `frag_llr_size`, and `origin_call` in BED tables, VCF INFO tags (`FRAG_SHORT_R`, `FRAG_LLR`, `ORIGIN`), `<prefix>_qc.json` fragmentomics metrics with high leukocyte DNA warnings, and interactive HTML viewer tooltips and `FF (Y/A/X/Size)` badge.
- Standardized Parquet Arrow schema version to `1.0.0` (`SCHEMA_VERSION = "1.0.0"`). Legacy sample Parquet files generated without fragmentomic fields (`fragment_lengths`, `endpoint_count`, `frag_gc_sum`) and pre-1.0.0 reference files are intentionally unsupported, requiring re-extraction via `ccnv extract` and reference rebuilding via `ccnv reference`.
- Zensical documentation site hosted on GitHub Pages (`https://matthdsm.github.io/copynumber-cnv/`), with automated deployment in `.github/workflows/docs.yml`, MathJax 3 LaTeX math support, and pixi tasks (`pixi run docs-build`, `pixi run docs-serve`).
- Codified documentation policy in `AGENTS.md` requiring all documentation amendments to be integrated into `docs/` and tested against `zensical build --clean --strict`.
- Core pipeline subcommands: `ccnv extract` (stream BAM/CRAM into sample Parquet with deterministic read QC and canonical contig filtering), `ccnv reference` (build cohort or flat uniform-depth FASTA models with Gavish-Donoho SVHT PCA and signal-subspace `KDTree` neighbor search), `ccnv ref-qc` (audit reference usability with `PASS`/`WARN`/`FAIL` heuristics), and `ccnv predict` (robust Huber PCA projection, weighted `Binseg`/`KernelCPD` segmentation, autocorrelation-corrected Stouffer Z-score calling).
- Cell-free DNA (cfDNA) prenatal analysis mode (`--mode prenatal`): non-negative Elastic Net 5 kb `chrY` `FFY`, convex logit-link `RidgeCV`/`ElasticNetCV` 100 kb `FFA`, Huber `FFX`, 2D stoichiometric Gaussian Bayes sex chromosome aneuploidy classifier with Mahalanobis $\chi^2$ outlier gate, precision-fused `active_ff`, and $\mathrm{LLR}_{\text{fetal}}$ segment discordance scoring (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`).
- Covariate normalization and robust depth scaling: joint natural cubic B-spline regression with interaction terms (`SplineTransformer` + `HuberRegressor`) for GC content and mappability bias correction (`--gc`, `--mappability`, `--min-mappability`), Smyth Empirical Bayes moderated variance shrinkage (`squeeze_var`), and DESeq2 median-of-ratios library size factor normalization.
- Deliverables and interactive reporting: bgzipped and tabix-indexed BED (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`) and standard VCFv4.2 outputs (`_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`), structured QC payloads (`<prefix>_qc.json`), and standalone interactive Plotly WebGL HTML viewer (`_plot.html`) with pan/zoom, cytoband search, and prenatal discordance overlays.
- Native performance and storage architecture: versioned Parquet storage schema (`1.0.0`) with Zstandard compression and Delta Binary Packing, Polars DataFrame expressions with zero-copy Arrow interop, mass-conserving 1D count rebinning, multi-threaded C-compiled `scipy.spatial.KDTree` neighbor search, and organism-agnostic contig handling.
- Documentation, benchmarks, and CI automation: mathematical methodology specification with 51 citations (`docs/methodology.md`), WisecondorX algorithmic comparison and runtime benchmarks (`docs/benchmark.md`), pipeline flowcharts (`docs/pipeline.md`), and GitHub Actions workflow building multi-arch container images (`ghcr.io/matthdsm/copynumber-cnv`).
