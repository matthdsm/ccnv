# `ccnv.features`

## Overview

The `ccnv.features` package implements in-memory feature extraction, genomic binning and rebinning, cohort validation and 1 Mb super-block DESeq2 depth normalization, Poisson covariate bias correction, cohort sex classification, cell-free DNA fragmentomic transformations, and non-invasive prenatal (NIPT) reference and inference workflows (`FFY`, `FFA`, `FFX`, `FF_size`, 4D erfc ray-marginalized SCA, reference-side 24-band Fisher read weighting, Student-$`t_4`$ IRLS active FF fusion, and 5-state self-referenced origin engine).

## Interface

- [`binning.py:bin_intervals`](./binning.py): Generates 0-based half-open interval starts and ends for a contig length and bin width.
- [`binning.py:bin_counts`](./binning.py): Aggregates 1D read start coordinates into `int64` bin counts via `np.bincount`.
- [`binning.py:bin_contig_fragments`](./binning.py): Accumulates 7-channel per-bin arrays (`count`, `paired_count`, `frag_len_sum`, `short_frag_count`, `endpoint_count`, `frag_gc_sum`, `frag_bands`) from streamed alignment templates.
- [`binning.py:rebin_counts`](./binning.py): Projects 1D count arrays across bin grids over matching contig lengths via cumulative knot interpolation.
- [`rebinning.py:rebin_sample_sketch`](./rebinning.py) & [`rebin_sample_buffers`](./rebinning.py): Rebin sample Arrow sketches and arrays in memory, using a Polars fast path on contained integer multiples and count-only interpolation on non-nested grids.
- [`cohort.py:Cohort`](./cohort.py): Synchronized cohort matrix container (`contigs`, `contig_lengths`, `starts`, `ends`, `counts`, `tables`).
- [`cohort.py:validate_cohort`](./cohort.py): Validates contig layouts, coordinate grids, and homogeneous sequencing across cohort sample tables.
- [`cohort.py:cohort_from_fasta`](./cohort.py): Builds a synthetic single-sample uniform-depth `Cohort` (`count = end - start`) from `(name, length)` pairs.
- [`cohort.py:estimate_superblock_size_factors`](./cohort.py), [`estimate_size_factors`](./cohort.py), & [`normalize_depth`](./cohort.py): Compute DESeq2 median-of-ratios library size factors over 1 Mb super-blocks.
- [`cohort.py:mask_low_coverage_per_contig`](./cohort.py) & [`mask_low_coverage`](./cohort.py): Flag usable bins per contig exceeding 5% of positive median contig coverage.
- [`sex.py:SexInference`](./sex.py) & [`SexLabel`](./sex.py): Cohort sex classification result and enumeration types.
- [`sex.py:infer_germline_sex_2d`](./sex.py) & [`infer_sample_sexes`](./sex.py): Classify germline `XX` vs. `XY` cohort samples via 2D non-PAR dosage separation.
- [`covariates.py:validate_min_mappability`](./covariates.py): Validates `min_mappability` is finite within $`[0.0, 1.0]`$.
- [`covariates.py:mask_covariates`](./covariates.py): Combines FASTA `N`-fraction, non-finite/out-of-range GC, and mappability threshold masks.
- [`covariates.py:fit_poisson_covariate_trends`](./covariates.py) & [`covariate_features`](./covariates.py): Remove multiplicative GC, mappability, and fragmentomic bias per sample via zero-inclusive `PoissonRegressor` regression on 200 bp GC quantiles and standardize active covariates.
- [`normalization.py:normalize_sample_branch`](./normalization.py) & [`normalize_branch`](./normalization.py): Project sample coverage onto reference PCA loadings via `HuberRegressor`, evaluate within-sample Prototype Graph baselines, and synthesize depth-decomposed Empirical Bayes precision weights.
- [`fragmentomics.py:extract_bin_fragmentomic_covariates`](./fragmentomics.py), [`extract_sample_fragmentomics`](./fragmentomics.py), [`is_paired_sample_metadata`](./fragmentomics.py), & [`resolve_fragmentomics_compatibility`](./fragmentomics.py): Computes per-bin fragmentomic metrics, decodes `FragmentomicsMetadata`, and checks compatibility.
- [`fragmentomics.py:fit_cohort_fragmentomics_model`](./fragmentomics.py) & [`apply_prenatal_fisher_weighting`](./fragmentomics.py): Deconvolve cohort fetal/maternal length densities across paired-end samples, apply reference-side 24-band Fisher weights, and fit the 36-feature logit-link `FF_size` ensemble.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py): Fits 5 kb `chrY` closed-form Poisson MSY `FFY`, coarsened 100 kb `FFA`, forward $`R_X`$, and paired-end fragmentomics models into `prenatal_autosomal` and `prenatal_ffy` reference branches.
- [`prenatal.py:predict_prenatal`](./prenatal.py), [`classify_sca`](./prenatal.py), & [`fuse_active_fetal_fraction`](./prenatal.py): Evaluates fetal sex, `FFY`/`FFA`/`FFX`/`FF_size`, 4D erfc ray-marginalized SCA calls, and Student-$`t_4`$ IRLS fused `active_ff`.
- [`prenatal_origin.py:annotate_prenatal_origin_and_lod`](./prenatal_origin.py) & [`annotate_prenatal_segments`](./prenatal_origin.py): Evaluate stoichiometric ratio/implied-FF discordance (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`) and 5-state self-referenced origin calls (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$).

## Invariants

- [`binning.py:rebin_counts`](./binning.py): Source and target partitions must share identical positive contig lengths and strictly increasing non-overlapping intervals; output counts are non-negative and conserve total read mass.
- [`cohort.py:validate_cohort`](./cohort.py): Rejects mixed paired-end and single-end cohorts (`ReferenceError`) as well as mismatched contig layouts or bin coordinates.
- [`covariates.py:fit_poisson_covariate_trends`](./covariates.py): Operates on usable bins; multiplicative per-sample trends are clamped to $`[0.20, 5.0]`$.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py): Requires at least one sample table, gracefully handling single-sex or unlabeled-FF cohorts via stoichiometric fallbacks.

## Side Effects

- None

## Verification

- Run: `pixi run test`
