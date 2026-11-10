# `ccnv.stats`

## Overview

The `ccnv.stats` package provides pure, stateless numerical and machine learning kernels for `ccnv`: deterministic SVD/PCA with Gavish-Donoho Singular Value Hard Thresholding (SVHT), low-dimensional signal-subspace `KDTree` off-contig nearest-bin search, Smyth Empirical Bayes variance moderation (`squeeze_var`), quantile B-spline `HuberRegressor` bias modeling, weighted change-point segmentation and AR(1)-adjusted Stouffer Z-score calling, multi-resolution prenatal estimators (`FFY`, `FFA`, `FFX`), and cell-free DNA fragmentomics kernels (`FF_size`, density deconvolution, Fisher-optimal read weights, and segment ``$`\mathrm{LLR}_{\text{size}}`$``).

## Interface

- [`pca.py:fit_pca`](./pca.py), [`optimal_svht_rank`](./pca.py), & [`pca_irlba_numpy`](./pca.py): Centered SVD decomposition with Gavish-Donoho SVHT rank selection and deterministic `svd_flip` sign conventions.
- [`neighbors.py:build_signal_space_features`](./neighbors.py) & [`select_nearest_bins`](./neighbors.py): Assemble ``$`\mathbb{R}^{P+C}`$`` signal-subspace feature matrices and execute per-contig leave-one-contig-out `scipy.spatial.KDTree` nearest-bin queries.
- [`neighbors.py:compute_null_models`](./neighbors.py), [`squeeze_var`](./neighbors.py), & [`distance_cutoff`](./neighbors.py): Evaluate PCA degrees-of-freedom-adjusted null variances, Smyth (2004) Empirical Bayes inverse-``$`\chi^2`$`` shrinkage (`brentq`), and 3-pass iterative ``$`\mu + 3\sigma`$`` squared-distance outlier cutoffs.
- [`regression.py:build_1d_basis`](./regression.py) & [`build_joint_covariate_basis`](./regression.py): Construct 1D or joint quantile B-spline design matrices (`SplineTransformer`) with standardized interaction terms across GC, mappability, mean fragment length, and cleavage endpoint ratio.
- [`regression.py:fit_sample_trend`](./regression.py) & [`apply_1d_correction`](./regression.py): Fit linear median-scaled `HuberRegressor` (`epsilon = 1.35`, `alpha = 1e-4`, `lstsq` fallback) trends clamped to ``$`[0.20, 5.0]`$`` and divide out multiplicative bias while preserving usable sample depth.
- [`segmentation.py:WeightedL2Cost`](./segmentation.py), [`segment_contig`](./segmentation.py), & [`segment_genome`](./segmentation.py): Prefix-sum ``$`O(1)`$`` weighted SSE Ruptures cost and pre-split euchromatic change-point segmentation (`Binseg`, `Pelt`, and unweighted `KernelCPD`).
- [`segmentation.py:compute_contig_rho1`](./segmentation.py) & [`evaluate_stouffer_segment_scores`](./segmentation.py): Estimate diploid-clipped (`|Z| <= 2.5`) contig lag-1 autocorrelation clamped to ``$`[0.0, 0.95]`$`` and evaluate AR(1)-adjusted Stouffer segment Z-scores.
- [`prenatal_math.py:infer_sex_gmm_1d`](./prenatal_math.py): 1D two-component heteroscedastic GMM with quartile initialization and `brentq` Bayes boundary solving on `chrY` read fractions.
- [`prenatal_math.py:fit_elasticnet_ffy`](./prenatal_math.py), [`fit_ffa_logit_ensemble`](./prenatal_math.py), [`rlm_huber`](./prenatal_math.py), & [`ols_fit`](./prenatal_math.py): Non-negative `ElasticNetCV` `chrY` `FFY` solver, convex logit-link `RidgeCV` + `ElasticNetCV` `FFA` ensemble with nested `KFold` RMSE, Huber RLM (`epsilon = 1.345`) for `FFX`, and legacy `fit_lasso_fista` / `fit_rprop_mlp` / `train_olm` helpers.
- [`fragmentomics.py:encode_kmer_4`](./fragmentomics.py), [`compute_motif_diversity_score`](./fragmentomics.py), [`compute_helical_periodicity`](./fragmentomics.py), [`estimate_sample_landmarks`](./fragmentomics.py), & [`align_histogram`](./fragmentomics.py): Base-4 ``$`5'`$`` 4-mer encoding, normalized Shannon MDS, ``$`10.4\text{ bp}`$`` DFT cleavage amplitude, parabolic chromatosome landmarking, and sub-bin linear histogram alignment.
- [`fragmentomics.py:deconvolve_fetal_maternal_densities`](./fragmentomics.py) & [`fisher_optimal_weights`](./fragmentomics.py): Cohort deconvolution of pure fetal ``$`\hat{p}_F(L)`$`` and maternal ``$`\hat{p}_M(L)`$`` length densities and 25-step bisection solver for depth-conserving Fisher read weights ``$`w^{\ast}(L) \in [0.25, 3.0]`$``.
- [`fragmentomics.py:build_fragmentomic_feature_vector`](./fragmentomics.py), [`fit_ff_fragmentomics_ensemble`](./fragmentomics.py), [`predict_ff_fragmentomics`](./fragmentomics.py), & [`segment_fragment_llr`](./fragmentomics.py): 36-feature standardized logit-link `FF_size` ensemble and per-segment stoichiometric mixture ``$`\mathrm{LLR}_{\text{size}}`$`` scorer.

## Invariants

- [`pca.py:optimal_svht_rank`](./pca.py) & [`fit_pca`](./pca.py): Select rank in ``$`[1, \min(P_{\max}, S, U)]`$`` with deterministic singular-vector sign orientation.
- [`neighbors.py:select_nearest_bins`](./neighbors.py): Never pairs bins located on the same contig.
- [`neighbors.py:squeeze_var`](./neighbors.py): Evaluates unclipped bias-centered log-variance moments over finite variances ``$`\gt 10^{-16}`$``, capping prior degrees of freedom at ``$`10^6`$`` when excess variance ``$`\le 10^{-7}`$``.
- [`segmentation.py:segment_contig`](./segmentation.py): Requires sorted non-overlapping intervals (``$`\text{start}_{i+1} \ge \text{end}_i`$``), positive finite `penalty`, and ``$`\text{jump} \ge 1`$``.
- [`fragmentomics.py:fisher_optimal_weights`](./fragmentomics.py): Enforces ``$`\mathbb{E}_{L \sim p_{\text{mix}}}[w^{\ast}(L)] = 1.0`$`` within ``$`[0.25, 3.0]`$`` at target fetal fraction ``$`f_0 = 0.10`$``.

## Side Effects

- [`neighbors.py:select_nearest_bins`](./neighbors.py): Executes multi-threaded `scipy.spatial.KDTree.query` worker threads when `threads > 1`.

## Verification

- Run: `pixi run test`
