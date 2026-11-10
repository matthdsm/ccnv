# `ccnv.features`

## Overview

The `ccnv.features` package implements in-memory feature extraction, 4-channel genomic binning and rebinning, cohort validation and DESeq2 depth normalization, joint quantile B-spline covariate bias correction, cell-free DNA fragmentomic transformations, and non-invasive prenatal (NIPT) reference and inference workflows (`FFY`, `FFA`, `FFX`, `FF_size`, 2D Bayesian SCA, Fisher read weighting, and segment origin scoring).

## Interface

- [`binning.py:bin_intervals`](./binning.py): Generates 0-based half-open interval starts and ends for a contig length and bin width.
- [`binning.py:bin_counts`](./binning.py): Aggregates 1D read start coordinates into `int64` bin counts via `np.bincount`.
- [`binning.py:bin_contig_fragments`](./binning.py): Accumulates 4-channel per-bin arrays (`count`, `fragment_lengths`, `endpoint_count`, `frag_gc_sum`) from streamed alignment templates.
- [`binning.py:rebin_counts`](./binning.py): Projects 1D count arrays across bin grids over matching contig lengths via cumulative knot interpolation (with overlap-scan fallback for gapped source bins).
- [`cohort.py:Cohort`](./cohort.py), [`SexInference`](./cohort.py), [`SexLabel`](./cohort.py): Synchronized cohort matrix container (`contigs`, `contig_lengths`, `starts`, `ends`, `counts`, `tables`) and sex-inference summary types.
- [`cohort.py:validate_cohort`](./cohort.py): Validates contig layouts, coordinate grids, and homogeneous paired-end vs. single-end sequencing across cohort sample tables (rebins in memory when `target_bin_size` is provided).
- [`cohort.py:rebin_sample_table`](./cohort.py) & [`rebin_cohort`](./cohort.py): Rebins sample Arrow tables in memory, using a 4-channel Polars coarsening fast-path on contained integer multiples and count-only interpolation (clearing fragmentomic channels and `ccnv.fragmentomics` metadata with a warning) on non-nested grids.
- [`cohort.py:cohort_from_fasta`](./cohort.py): Builds a synthetic single-sample uniform-depth `Cohort` (`count = end - start`) from `(name, length)` pairs.
- [`cohort.py:estimate_size_factors`](./cohort.py) & [`normalize_depth`](./cohort.py): Computes DESeq2 median-of-ratios library size factors and scales sample counts while preserving median usable sample depth.
- [`cohort.py:mask_low_coverage`](./cohort.py): Flags usable bins whose summed normalized cohort depth exceeds 5% of the positive median bin sum.
- [`cohort.py:infer_sample_sexes`](./cohort.py): Classifies germline `XX` vs. `XY` cohort samples via `chrY` read fractions and order-statistic gap-MAD separation.
- [`covariates.py:validate_min_mappability`](./covariates.py): Validates `min_mappability` is finite within ``$`[0.0, 1.0]`$``.
- [`covariates.py:mask_covariates`](./covariates.py): Combines FASTA `N`-fraction, non-finite/out-of-range GC, and mappability threshold masks.
- [`covariates.py:covariate_features`](./covariates.py): Standardizes active covariates (`gc`, `mappability`, `mean_frag_len`, `endpoint_ratio`) via population z-scores for signal-subspace neighbor search.
- [`covariates.py:correct_covariate_bias`](./covariates.py): Removes multiplicative GC, mappability, and fragmentomic bias per sample via joint quantile B-spline `HuberRegressor` trends (`build_joint_covariate_basis` + `apply_1d_correction`).
- [`fragmentomics.py:evaluate_weighted_coverage`](./fragmentomics.py), [`apply_prenatal_fisher_weighting`](./fragmentomics.py), & [`align_reference_to_weighted_sample`](./fragmentomics.py): Applies Fisher-optimal fragment length weights ``$`w^{\ast}(L)`$`` via a 65,536-entry lookup table and aligns reference `pca_mean` and `null_mean` baselines.
- [`fragmentomics.py:extract_bin_fragmentomic_covariates`](./fragmentomics.py) & [`extract_sample_fragmentomics`](./fragmentomics.py): Computes per-bin mean fragment length and cleavage endpoint ratios, and decodes `FragmentomicsMetadata` from Arrow schema metadata.
- [`fragmentomics.py:fit_cohort_fragmentomics_model`](./fragmentomics.py): Deconvolves cohort fetal/maternal length densities across all paired-end samples and fits the 36-feature logit-link `FF_size` ensemble when at least 5 samples carry labeled `ff` values.
- [`fragmentomics.py:fuse_active_fetal_fraction`](./fragmentomics.py), [`extract_segment_fragment_lengths`](./fragmentomics.py), & [`score_segment_origin`](./fragmentomics.py): Performs Bates-Granger `active_ff` precision fusion and scores segment short-fragment ratios, ``$`\mathrm{LLR}_{\text{size}}`$``, and `origin_call`.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py): Fits 5 kb `chrY` `FFY`, coarsened 100 kb `FFA` and `FFX`, and paired-end fragmentomics models into `prenatal_autosomal` and `prenatal_ffy` reference branches plus `PrenatalModelMetadata`.
- [`prenatal.py:predict_prenatal`](./prenatal.py), [`classify_sca`](./prenatal.py), & [`annotate_prenatal_segments`](./prenatal.py): Evaluates fetal sex, `FFY`/`FFA`/`FFX`/`FF_size`, 2D stoichiometric Bayesian SCA calls, `active_ff`, stoichiometric ratio/implied-FF discordance (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`), and fragmentomic origin calls.

## Invariants

- [`binning.py:rebin_counts`](./binning.py): Source and target partitions must share identical positive contig lengths and strictly increasing non-overlapping intervals; output counts are non-negative and conserve total read mass.
- [`cohort.py:validate_cohort`](./cohort.py): Rejects mixed paired-end and single-end cohorts (`ReferenceError`) as well as mismatched contig layouts or bin coordinates.
- [`covariates.py:correct_covariate_bias`](./covariates.py): Operates strictly on unmasked usable bins; multiplicative per-sample trends are clamped to ``$`[0.20, 5.0]`$``.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py): Requires at least one sample table and matching `entries` length, gracefully handling single-sex or unlabeled-FF cohorts via stoichiometric/theoretical fallbacks.

## Side Effects

- None (all transforms in `ccnv.features` operate purely in memory, emitting standard `logging` warnings when non-nested rebinning drops fragmentomic channels).

## Verification

- Run: `pixi run test`
