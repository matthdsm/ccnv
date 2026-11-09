# `ccnv.stats`

## Overview

The `ccnv.stats` package provides stateless numerical, statistical, and machine learning kernels for `ccnv`. It implements deterministic truncated Principal Component Analysis (PCA), batched off-contig nearest-neighbor Euclidean search, robust Huber iteratively reweighted least squares (IRLS) regression, weighted change-point segmentation using custom Ruptures cost functions, and non-negative Lasso and neural network optimization for prenatal fetal fraction estimation.

## Interface

- `[src/ccnv/stats/pca.py:fit_pca]`: Fits truncated SVD or IRLBA across sample depth matrices with deterministic sign conventions.
- `[src/ccnv/stats/pca.py:pca_irlba_numpy]`: Pure NumPy/SciPy implementation of implicitly restarted Lanczos bidiagonalization.
- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Parallelized off-contig nearest-bin selection in PCA or covariate loading space.
- `[src/ccnv/stats/neighbors.py:compute_null_models]`: Parameterizes per-bin empirical log2 ratio null expectations and standard deviations.
- `[src/ccnv/stats/neighbors.py:distance_cutoff]`: Computes empirical Euclidean distance percentiles for reference QC auditing.
- `[src/ccnv/stats/regression.py:build_1d_basis]`: Formulates standardized polynomial basis matrices for covariate modeling.
- `[src/ccnv/stats/regression.py:fit_sample_trend]`: Fits multiplicative bias trends via 3-iteration Huber IRLS regression.
- `[src/ccnv/stats/regression.py:apply_1d_correction]`: Divides out multiplicative bias trends and rescales depths to preserve totals.
- `[src/ccnv/stats/segmentation.py:WeightedL2Cost]`: Custom Ruptures `BaseCost` computing weighted SSE from weighted segment means.
- `[src/ccnv/stats/segmentation.py:segment_contig]`: Partitions a contig across assembly gaps and executes weighted Binary Segmentation.
- `[src/ccnv/stats/segmentation.py:segment_genome]`: Coordinates whole-genome change-point segmentation across all contigs.
- `[src/ccnv/stats/prenatal_math.py:fit_lasso_fista]`: Fits non-negative Lasso regression via coordinate descent (or `nnls` when $\alpha \le 0$) for `FFY` estimation.
- `[src/ccnv/stats/prenatal_math.py:fit_rprop_mlp]`: Trains a 2-unit Sigmoid multi-layer perceptron via quasi-Newton `L-BFGS-B` with Glorot initialization for `PREFACE`.
- `[src/ccnv/stats/prenatal_math.py:infer_sex_gmm_1d]`: 1D two-component Gaussian Mixture Model with Brent's root-finding classifying sample fetal sex.
- `[src/ccnv/stats/prenatal_math.py:ols_fit]`: Ordinary Least Squares linear regression kernel.
- `[src/ccnv/stats/prenatal_math.py:rlm_huber]`: Robust Linear Model regression with Huber M-estimation ($\epsilon = 1.345$) for `FFX`.
- `[src/ccnv/stats/prenatal_math.py:train_olm]`: Trains an Ordinary Linear Model for prenatal skew adjustment.

## Invariants

- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Nearest-bin search explicitly enforces infinite distance ($d_{ii} = \infty$) for all bins on the same contig to guarantee off-contig selection.
- `[src/ccnv/stats/neighbors.py:compute_null_models]`: Null distribution sampling is bounded to at most 100 cohort samples; non-finite standard deviations are coerced to 0.0.
- `[src/ccnv/stats/pca.py:fit_pca]`: Requires $n_{\text{components}} \ge 1$ and at least $n_{\text{components}}$ valid samples; enforces deterministic sign orientation on PCA loadings.
- `[src/ccnv/stats/regression.py:fit_sample_trend]`: 3-iteration Huber IRLS regression uses tuning threshold $\delta = \max(2.5 \cdot \sigma_{\text{MAD}}, 0.08)$ and weight floor $0.05$; trend is clamped to $[0.20, 5.0]$.
- `[src/ccnv/stats/segmentation.py:WeightedL2Cost]`: Signal values and weights must have identical lengths; non-finite values or negative weights are zeroed.
- `[src/ccnv/stats/segmentation.py:segment_contig]`: Bins must be sorted, non-overlapping ($\text{start}_{i+1} \ge \text{end}_i$), with finite positive penalty and $\text{jump} \ge 1$.
- `[src/ccnv/stats/prenatal_math.py:fit_lasso_fista]`: Non-negative Lasso enforces non-negative regression weights ($\boldsymbol{\beta} \ge \mathbf{0}$).

## Side Effects

- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Spawns worker process pool via `concurrent.futures.ProcessPoolExecutor` when bin count $> 2\,000$ and $\text{threads} > 1$.

## Verification

- Run: `pytest`
