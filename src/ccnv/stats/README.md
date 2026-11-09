# `ccnv.stats`

## Overview

The `ccnv.stats` package provides stateless numerical, statistical, and machine learning kernels for `ccnv`. It implements deterministic truncated Principal Component Analysis (PCA) with Gavish-Donoho Optimal Singular Value Hard Thresholding (SVHT), low-dimensional signal-subspace `KDTree` off-contig nearest-neighbor search, Smyth Empirical Bayes variance shrinkage (`squeeze_var`), joint natural cubic B-spline and Huber M-estimation regression, weighted change-point segmentation using custom Ruptures cost functions, non-negative Elastic Net (`FFY`), and convex logit-link regularized linear ensembles (`FFA`).

## Interface

- `[src/ccnv/stats/pca.py:fit_pca]`: Fits truncated SVD across sample depth matrices with Gavish-Donoho SVHT rank selection and deterministic sign conventions.
- `[src/ccnv/stats/pca.py:optimal_svht_rank]`: Evaluates Gavish-Donoho Optimal Singular Value Hard Thresholding to select optimal PCA subspace rank under unknown noise variance.
- `[src/ccnv/stats/pca.py:pca_irlba_numpy]`: Truncated PCA decomposition with deterministic sign alignment.
- `[src/ccnv/stats/neighbors.py:build_signal_space_features]`: Embeds PCA loadings and standardized sequence covariates into low-dimensional signal subspace $\mathbb{R}^{P+2}$.
- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Parallelized off-contig nearest-bin selection in signal subspace via C-compiled `scipy.spatial.KDTree`.
- `[src/ccnv/stats/neighbors.py:compute_null_models]`: Parameterizes per-bin empirical log2 ratio null expectations and degrees-of-freedom-corrected standard deviations.
- `[src/ccnv/stats/neighbors.py:distance_cutoff]`: Computes empirical Euclidean distance percentiles for reference QC auditing.
- `[src/ccnv/stats/neighbors.py:squeeze_var]`: Result-equivalent Smyth (2004) / Phipson et al. (2016) Empirical Bayes variance shrinkage under inverse-$\chi^2$ prior.
- `[src/ccnv/stats/regression.py:build_1d_basis]`: Formulates standardized polynomial basis matrices for covariate modeling.
- `[src/ccnv/stats/regression.py:build_joint_covariate_basis]`: Formulates joint natural cubic B-spline and interaction basis matrix for GC and mappability modeling.
- `[src/ccnv/stats/regression.py:fit_sample_trend]`: Fits multiplicative bias trends via robust Huber M-estimation or IRLS regression.
- `[src/ccnv/stats/regression.py:apply_1d_correction]`: Divides out multiplicative bias trends and rescales depths to preserve totals.
- `[src/ccnv/stats/segmentation.py:WeightedL2Cost]`: Custom Ruptures `BaseCost` computing weighted SSE from weighted segment means in $O(1)$ time via prefix sums.
- `[src/ccnv/stats/segmentation.py:segment_contig]`: Partitions a contig across euchromatic blocks and executes change-point segmentation (KernelCPD, Pelt, or Binseg).
- `[src/ccnv/stats/segmentation.py:segment_genome]`: Coordinates whole-genome change-point segmentation across all contigs.
- `[src/ccnv/stats/prenatal_math.py:fit_elasticnet_ffy]`: Fits non-negative ElasticNetCV regression across euchromatic `chrY` bins for `FFY` estimation.
- `[src/ccnv/stats/prenatal_math.py:fit_lasso_fista]`: Fits non-negative Lasso regression via coordinate descent for `FFY` estimation.
- `[src/ccnv/stats/prenatal_math.py:fit_ffa_logit_ensemble]`: Calibrates convex logit-link `RidgeCV` + `ElasticNetCV` ensemble for `FFA` (fetal fraction autosomal) prediction.
- `[src/ccnv/stats/prenatal_math.py:fit_rprop_mlp]`: Trains a 2-unit Sigmoid multi-layer perceptron via quasi-Newton `L-BFGS-B` with Glorot initialization for legacy autosomal models.
- `[src/ccnv/stats/prenatal_math.py:infer_sex_gmm_1d]`: 1D two-component Gaussian Mixture Model with Brent's root-finding classifying sample fetal sex.
- `[src/ccnv/stats/prenatal_math.py:ols_fit]`: Ordinary Least Squares linear regression kernel.
- `[src/ccnv/stats/prenatal_math.py:rlm_huber]`: Robust Linear Model regression with Huber M-estimation ($\epsilon = 1.345$) for `FFX`.
- `[src/ccnv/stats/prenatal_math.py:train_olm]`: Trains an Ordinary Linear Model for prenatal skew adjustment.

## Invariants

- `[src/ccnv/stats/pca.py:optimal_svht_rank]`: Selects rank between 1 and $\min(P_{\max}, S, U)$ via closed-form Gavish-Donoho threshold polynomial.
- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Nearest-bin search explicitly queries off-contig bins via leave-one-contig-out `scipy.spatial.KDTree` to strictly prevent intra-contig neighbor selection.
- `[src/ccnv/stats/neighbors.py:squeeze_var]`: Requires finite positive degrees of freedom and at least 3 positive variance estimates; tail variances Winsorized at `(0.05, 0.10)`.
- `[src/ccnv/stats/neighbors.py:compute_null_models]`: Null distribution sampling is bounded to at most 100 cohort samples; residual degrees of freedom are adjusted for PCA rank.
- `[src/ccnv/stats/pca.py:fit_pca]`: Requires $n_{\text{components}} \ge 1$ and at least $n_{\text{components}}$ valid samples; enforces deterministic sign orientation on PCA loadings.
- `[src/ccnv/stats/regression.py:fit_sample_trend]`: Huber regression uses tuning threshold $\epsilon = 1.35$ and penalty $\alpha = 10^{-4}$; trend is clamped to $[0.20, 5.0]$.
- `[src/ccnv/stats/segmentation.py:WeightedL2Cost]`: Signal values and weights must have identical lengths; non-finite values or negative weights are zeroed.
- `[src/ccnv/stats/segmentation.py:segment_contig]`: Bins must be sorted, non-overlapping ($\text{start}_{i+1} \ge \text{end}_i$), with finite positive penalty and $\text{jump} \ge 1$.
- `[src/ccnv/stats/prenatal_math.py:fit_elasticnet_ffy]`: Non-negative Elastic Net enforces non-negative regression weights ($\mathbf{w} \ge \mathbf{0}$) with out-of-fold cross-validation.

## Side Effects

- `[src/ccnv/stats/neighbors.py:select_nearest_bins]`: Queries C-compiled `scipy.spatial.KDTree` using multi-threaded execution (`workers=threads`).

## Verification

- Run: `pytest`
