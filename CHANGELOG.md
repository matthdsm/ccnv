# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

## [Unreleased]

### Added

- Cell-free DNA (cffDNA vs maternal cfDNA) fragmentomic feature extraction and size distribution modeling across `extract`, `reference`, and `predict`:
  - `ccnv extract`: Single-pass alignment streaming with paired-end dyad midpoint binning (`TLEN > 0`), extracting clamped insert lengths (``$`30\text{–}600\text{ bp}`$``), unclipped ``$`5'`$`` 4-mer motif codes, fragment GC content, and cleavage endpoint counts alongside `ccnv.fragmentomics` Parquet metadata.
  - Parabolic chromatosome landmarking and sub-bin linear histogram alignment (``$`\Delta_{\text{bp}} = M_{\text{chrom}} - 166.0 \in [-11.0, +14.0]\text{ bp}`$``) standardizing platform sizing offsets across cohorts.
  - 36-feature convex logit-link `FF_size` ensemble (`RidgeCV` + `ElasticNetCV`) incorporating 28 mode-aligned `5 bp` log-density bands, core-to-chromatosome and sub-nucleosomal peak ratios, ``$`10.4\text{ bp}`$`` helical DFT cleavage amplitude, Motif Diversity Score (MDS), and top nuclease 4-mer frequencies (`CCCA`, `CCAG`, `CCTG`).
  - Cohort fragment length deconvolution estimating pure fetal ``$`\hat{p}_F(L)`$`` (pooled male `chrY` minus female-scaled misalignment background) and maternal ``$`\hat{p}_M(L)`$`` (autosomal zero-fetal-fraction OLS intercept) probability densities.
  - Depth-conserving Fisher-optimal read weighting ``$`w^{\ast}(L) \in [0.25, 3.0]`$`` solved via 25-step bisection under ``$`\mathbb{E}_{L \sim p_{\text{mix}}}[w^{\ast}(L)] = 1.0`$``, paired with weighted-reference baseline alignment during `ccnv predict --mode prenatal`.
  - Contained-interval 4-channel Polars coarsening (`count`, `fragment_lengths`, `endpoint_count`, `frag_gc_sum`) with mass-conserving 1D count interpolation fallback on non-nested bin grids.
  - Augmented Huber B-spline covariate regression and ``$`\mathbb{R}^{P+C}`$`` (`C` up to 4) `KDTree` off-contig neighbor matching incorporating per-bin mean fragment length ``$`\bar{L}_i`$`` and cleavage endpoint ratio ``$`E_i / \max(C_i, 1.0)`$``.
  - Multi-modal Bates-Granger inverse-variance precision fusion combining concordant `FFY`, `FFA`, `FFX`, and `FF_size` estimates into `active_ff`.
  - Segment-level fragmentomic log-likelihood ratio scoring (``$`\mathrm{LLR}_{\text{size}}`$``) and biological origin classification (`FETAL_CONFIRMED`, `MATERNAL_CONFIRMED`, `INDETERMINATE`).
  - Polymorphic alignment ingestion in `ccnv predict`: directly accepts `.bam`, `.cram` (with `--fasta`), and `.parquet`/`.ccnv` sample files.
  - Deliverables: annotated `frag_short_ratio`, `frag_llr_size`, and `origin_call` in BED tables, prenatal VCF header/INFO/FORMAT fields (`EXP_LOG2R`, `IMPLIED_FF`, `DISCORDANT`, `DISC_REASON`, `FRAG_SHORT_R`, `FRAG_LLR`, `ORIGIN`, `FF`), `<prefix>_qc.json` fragmentomics metrics with high leukocyte DNA warnings, and interactive HTML viewer tooltips and `FF (Y/A/X/Size)` badge.
- Standardized Parquet Arrow schema version to `1.0.0` (`SCHEMA_VERSION = "1.0.0"`). Legacy sample Parquet files generated without fragmentomic fields (`fragment_lengths`, `endpoint_count`, `frag_gc_sum`) and pre-1.0.0 reference files are intentionally unsupported, requiring re-extraction via `ccnv extract` and reference rebuilding via `ccnv reference`.
- Zensical documentation site hosted on GitHub Pages (`https://matthdsm.github.io/copynumber-cnv/`), with automated deployment in `.github/workflows/docs.yml`, MathJax 3 LaTeX math support, guarded instant-navigation re-typesetting (`docs/javascripts/mathjax.js`), and pixi tasks (`pixi run docs-build`, `pixi run docs-serve`).
- Dual-target Markdown and LaTeX rendering rules in `AGENTS.md` and automated contract checks in `tests/test_docs_markdown.py`, enforcing Zensical `pymdownx.arithmatex` + admonition compatibility across `docs/**/*.md` and GitHub `cmark-gfm` compatibility across repository `README.md`, `CHANGELOG.md`, and package documentation.
- Core pipeline subcommands: `ccnv extract` (stream BAM/CRAM into sample Parquet with deterministic read QC and canonical contig filtering), `ccnv reference` (build cohort or flat uniform-depth FASTA models with Gavish-Donoho SVHT PCA and signal-subspace `KDTree` neighbor search), `ccnv ref-qc` (audit reference usability with `PASS`/`WARN`/`FAIL` heuristics), and `ccnv predict` (robust Huber PCA projection, weighted `Binseg`/`Pelt` and unweighted `KernelCPD` segmentation, autocorrelation-corrected Stouffer Z-score calling).
- Cell-free DNA (cfDNA) prenatal analysis mode (`--mode prenatal`): non-negative Elastic Net 5 kb `chrY` `FFY`, convex logit-link `RidgeCV`/`ElasticNetCV` 100 kb `FFA`, Huber `FFX`, 2D stoichiometric Gaussian Bayes sex chromosome aneuploidy classifier with Mahalanobis ``$`\chi^2`$`` outlier gate, precision-fused `active_ff`, and stoichiometric ratio/implied-FF segment discordance annotation (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`).
- Covariate normalization and robust depth scaling: joint quantile B-spline regression with interaction terms (`SplineTransformer` + `HuberRegressor`) for GC content, mappability, and fragmentomic bias correction (`--gc`, `--mappability`, `--min-mappability`), Smyth Empirical Bayes moderated variance shrinkage (`squeeze_var`), and DESeq2 median-of-ratios library size factor normalization.
- Deliverables and interactive reporting: bgzipped and tabix-indexed BED (`_bins.bed.gz`, `_segments.bed.gz`, `_calls.bed.gz`) and standard VCFv4.2 outputs (`_bins.vcf.gz`, `_segments.vcf.gz`, `_calls.vcf.gz`), structured QC payloads (`<prefix>_qc.json`), and standalone interactive Plotly WebGL HTML viewer (`_plot.html`) with pan/zoom, cytoband search, and prenatal discordance overlays.
- Native performance and storage architecture: versioned Parquet storage schema (`1.0.0`) with Zstandard compression and Delta Binary Packing, Polars DataFrame expressions with zero-copy Arrow interop, mass-conserving count rebinning, multi-threaded C-compiled `scipy.spatial.KDTree` neighbor search, and organism-agnostic contig handling.
- Documentation and CI automation: mathematical methodology specification with 42 peer-reviewed citations (`docs/methodology.md`), pipeline flowcharts (`docs/pipeline.md`), and GitHub Actions workflow building multi-arch container images (`ghcr.io/matthdsm/copynumber-cnv`).
