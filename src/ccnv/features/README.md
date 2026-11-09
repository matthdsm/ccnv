# `ccnv.features`

## Overview

The `ccnv.features` package provides feature extraction, dynamic genomic rebinning, cohort depth normalization, covariate bias correction, and non-invasive prenatal (NIPT) modeling workflows. It handles coordinate transformations between heterogeneous bin sizes, robust joint B-spline bias estimation against GC content and mappability, and multi-resolution fetal fraction calibration (`FFY`, `PREFACE`, `FFX`).

## Interface

- `[src/ccnv/features/binning.py:bin_counts]`: Bins raw read 5' coordinates into 1D integer arrays.
- `[src/ccnv/features/binning.py:bin_intervals]`: Generates 0-based half-open interval starts and ends for a contig.
- `[src/ccnv/features/binning.py:rebin_counts]`: Projects counts across coordinate systems via mass-conserving cumulative knot interpolation.
- `[src/ccnv/features/cohort.py:Cohort]`: Synchronized cohort container holding multi-sample counts, contig metadata, and sample identifiers.
- `[src/ccnv/features/cohort.py:rebin_sample_table]`: Rebins an Apache Arrow sample table in memory, preserving continuous `float64` counts with a Rust-parallelized Polars integer coarsening fast-path.
- `[src/ccnv/features/cohort.py:cohort_from_fasta]`: Synthesizes an artificial flat uniform-depth cohort from FASTA sequence lengths.
- `[src/ccnv/features/cohort.py:estimate_size_factors]`: Calculates DESeq2 median-of-ratios library size factors across cohort samples with a 50% breakdown point.
- `[src/ccnv/features/cohort.py:normalize_depth]`: Normalizes per-sample bin depths using DESeq2 median-of-ratios scaling (or positive-median/total-sum fallback).
- `[src/ccnv/features/cohort.py:mask_low_coverage]`: Masks bins with aggregate cohort depth at or below 5% of median bin depth.
- `[src/ccnv/features/cohort.py:infer_sample_sexes]`: Infers genetic sex (`XX` vs `XY`) across cohort samples via gonosome depth ratios and order-statistic gap-MAD separation.
- `[src/ccnv/features/cohort.py:rebin_cohort]`: Parallelized cohort-wide rebinning to target bin coordinates.
- `[src/ccnv/features/covariates.py:validate_min_mappability]`: Validates that mappability thresholds fall within `[0.0, 1.0]`.
- `[src/ccnv/features/covariates.py:covariate_features]`: Formulates standardized polynomial basis matrices for GC and mappability tracks.
- `[src/ccnv/features/covariates.py:mask_covariates]`: Masks bins failing valid base count, ambiguous base fraction, or mappability criteria.
- `[src/ccnv/features/covariates.py:correct_covariate_bias]`: Normalizes GC and mappability biases using joint natural cubic B-spline and Huber M-estimation regression.
- `[src/ccnv/features/prenatal.py:fit_prenatal_models]`: Calibrates multi-resolution prenatal reference models (`FFY`, `PREFACE`, `FFX`).
- `[src/ccnv/features/prenatal.py:predict_prenatal]`: Evaluates active fetal fraction via Bates-Granger precision fusion, fetal sex, and sex chromosome aneuploidies.
- `[src/ccnv/features/prenatal.py:classify_sca]`: Classifies sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`) via a 2D stoichiometric Gaussian Bayes model with Mahalanobis $\chi^2$ gate.
- `[src/ccnv/features/prenatal.py:annotate_prenatal_segments]`: Annotates CNV segments with expected prenatal ratios, implied fetal fractions, and LLR discordance flags.

## Invariants

- `[src/ccnv/features/binning.py:rebin_counts]`: Bins must share matching contig lengths and monotonically increasing coordinates; counts are strictly non-negative and total read mass is conserved.
- `[src/ccnv/features/cohort.py:validate_cohort]`: Input cohort sample tables must share identical contig layouts, identical contig lengths, and matching bin coordinates.
- `[src/ccnv/features/cohort.py:rebin_sample_table]`: Outputs `float64` counts, utilizing a Polars integer fast-path when target bin size is an exact multiple of source bin size.
- `[src/ccnv/features/covariates.py:validate_min_mappability]`: Minimum mappability threshold must be a finite float in $[0.0, 1.0]$.
- `[src/ccnv/features/covariates.py:correct_covariate_bias]`: Bins with non-positive valid bases, ambiguous base fraction $N > 0.5$, or mappability below `--min-mappability` are masked; multiplicative bias trend is clamped to $[0.20, 5.0]$.
- `[src/ccnv/features/prenatal.py:fit_prenatal_models]`: Cohort must contain both female and male control samples to calibrate gonosome models and prenatal reference branches.

## Side Effects

- `[src/ccnv/features/cohort.py:cohort_from_fasta]`: Generates FASTA index file (`.fai`) on disk via `pysam.faidx` when missing.
- `[src/ccnv/features/cohort.py:rebin_cohort]`: Spawns worker process pool via `concurrent.futures.ProcessPoolExecutor` when cohort contains multiple samples and threads $> 1$.

## Verification

- Run: `pytest`
