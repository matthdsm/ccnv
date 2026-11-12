# `ccnv.stats`

## Overview

The `ccnv.stats` package provides pure, stateless numerical and machine learning kernels for `ccnv`: deterministic SVD/PCA with Gavish-Donoho Singular Value Hard Thresholding (SVHT) on variance-whitened residuals, Prototype Graph ($`M \le 2{,}048`$) reference modeling, Smyth Empirical Bayes biological overdispersion moderation (`squeeze_var`, `bio_dispersion`), `PoissonRegressor` covariate modeling, calibrated `ruptures.Pelt` segmentation with modified BIC (mBIC) penalty, dyadic multi-scale BLUE Z-score aggregation, discrete integer copy-number Bayes factors ($`CN \in \lbrace 0, 1, 2, 3, 4+\rbrace`$), closed-form Poisson MSY `FFY`, forward $`R_X`$, 4D erfc ray-marginalized SCA, Student-$`t_4`$ IRLS active FF fusion, 24-band Fisher read weighting, and 5-state self-referenced origin classification.

## Interface

- [`pca.py:fit_whitened_pca`](./pca.py), [`pca_irlba_numpy`](./pca.py), [`project_huber_pca_wave`](./pca.py), & [`half_sample_mode`](./pca.py): Centered SVD decomposition on variance-whitened residuals with Gavish-Donoho SVHT rank selection, deterministic `svd_flip` sign conventions, robust `HuberRegressor` PCA wave projection, and recursive half-sample mode location estimation.
- [`neighbors.py:fit_prototype_graph`](./neighbors.py), [`compute_loco_prototype_support`](./neighbors.py), [`calibrate_prototype_dispersion`](./neighbors.py), & [`squeeze_var`](./neighbors.py): Build MiniBatch K-Means Prototype Graph ($`M \le 2{,}048`$) in standardized signal subspace, query $`k=4`$ nearest off-contig prototypes with soft inverse-distance weights, subtract Poisson shot-noise variance, and apply Smyth (2004) Empirical Bayes shrinkage to isolate depth-decomposed biological overdispersion (`bio_dispersion`).
- [`regression.py:build_exogenous_covariate_basis`](./regression.py), [`fit_poisson_trend`](./regression.py), & [`fit_logit_linear_ensemble`](./regression.py): Construct exogenous covariate design matrices (200 bp GC quantiles, mappability splines, cohort fragmentomics), fit zero-inclusive `PoissonRegressor` ($`\alpha = 10^{-4}`$) multiplicative trends, and fit convex logit-link `RidgeCV`/`ElasticNetCV` ensembles.
- [`segmentation.py:WeightedL2Cost`](./segmentation.py), [`ContigSegmentationInput`](./segmentation.py), [`segment_contig`](./segmentation.py), & [`segment_and_score_genome`](./segmentation.py): Prefix-sum $`O(1)`$ weighted SSE Ruptures cost, pre-split euchromatic change-point segmentation using calibrated `ruptures.Pelt` with mBIC penalty, contig-indexed dyadic multi-scale BLUE segment Z-scores ($`Z_{\text{BLUE}}`$), and discrete integer copy-number log-Bayes factors over $`CN \in \lbrace 0, 1, 2, 3, 4+\rbrace`$.
- [`prenatal_math.py:fit_poisson_msy_ffy`](./prenatal_math.py) & [`evaluate_poisson_msy_ffy`](./prenatal_math.py): Closed-form Poisson MLE `FFY` with analytical standard error propagation over clean MSY bins.
- [`prenatal_math.py:fit_forward_ffx_model`](./prenatal_math.py) & [`evaluate_forward_ffx`](./prenatal_math.py): Forward-calibrated $`R_X`$ baseline regression across `XX` ($`f=0`$) and `XY` ($`f=\hat{f}_{\text{FFY}}`$) controls alongside `FFA` logit-ensemble helpers.
- [`prenatal_math.py:classify_sca_ray_4d`](./prenatal_math.py) & [`fuse_active_fetal_fraction_student_t4`](./prenatal_math.py): Closed-form erfc Gaussian mixture ray-marginalized SCA classifier and Student-$`t_4`$ IRLS multi-modal active FF fusion.
- [`fragmentomics.py:compute_band_fisher_weights`](./fragmentomics.py), [`evaluate_band_weighted_counts`](./fragmentomics.py), & [`deconvolve_fetal_maternal_densities`](./fragmentomics.py): Reference-side 24-band Fisher read weights ($`w^{\ast}_{0.08}`$) and cohort deconvolution of pure fetal $`\hat{p}_F(L)`$ and maternal $`\hat{p}_M(L)`$ densities.
- [`fragmentomics.py:build_fragmentomic_feature_vector`](./fragmentomics.py), [`fit_ff_fragmentomics_ensemble`](./fragmentomics.py), & [`predict_ff_fragmentomics`](./fragmentomics.py): 36-feature standardized logit-link `FF_size` ensemble.

## Invariants

- [`pca.py:fit_whitened_pca`](./pca.py): Selects rank in $`[0, \min(P_{\max}, S - 2, U - 1)]`$ with deterministic singular-vector sign orientation.
- [`neighbors.py:fit_prototype_graph`](./neighbors.py): Never pairs bins with prototypes located on the same contig.
- [`segmentation.py:segment_contig`](./segmentation.py): Requires sorted non-overlapping intervals ($`\text{start}_{i+1} \ge \text{end}_i`$), positive finite penalty, and $`\text{jump} \ge 1`$.
- [`fragmentomics.py:compute_band_fisher_weights`](./fragmentomics.py): Enforces unit expectation under mixture density within $`[0.25, 3.0]`$ at target fetal fraction $`f_0 = 0.08`$.

## Side Effects

- None

## Verification

- Run: `pixi run test`
