# `ccnv.features`

## Overview

The `ccnv.features` package implements in-memory feature extraction, genomic binning and rebinning, cohort validation and 1 Mb super-block DESeq2 depth normalization, Poisson covariate bias correction, cohort sex classification, cell-free DNA fragmentomic transformations, and non-invasive prenatal (NIPT) reference and inference workflows (`FFY`, `FFA`, `FFX`, `FF_size`, 4D erfc ray-marginalized SCA, reference-side 24-band Fisher read weighting, Student-$`t_4`$ IRLS active FF fusion, and 5-state self-referenced origin engine).

## Interface

- [`binning.py:bin_intervals`](./binning.py), [`build_coordinate_grid`](./binning.py), [`compute_bin_exposure`](./binning.py), [`rebin_counts`](./binning.py), & [`rebin_counts_matrix`](./binning.py): Generate 0-based half-open interval partitions, fractional bin exposures, and mass-conserving 1D/2D cumulative knot rebinning.
- [`rebinning.py:rebin_sample_buffers`](./rebinning.py): Rebins `SampleBuffers` in memory, using a Polars fast path across all channels on contained integer multiples and count-only interpolation on non-nested grids.
- [`cohort.py:Cohort`](./cohort.py), [`load_and_assemble_cohort`](./cohort.py), [`validate_cohort`](./cohort.py), & [`rebin_cohort`](./cohort.py): Stream and validate synchronized Fortran-contiguous `float32` cohort matrices (spilling to `np.memmap` above `max_batch_memory_mb`).
- [`cohort.py:cohort_from_fasta`](./cohort.py): Builds a 0-sample coordinate grid from `(name, length)` pairs for flat FASTA reference models.
- [`cohort.py:build_fisher_weighted_cohort`](./cohort.py): Applies reference-side 24-band Fisher read weights ($`w^{\ast}_{0.08}`$) across cohort sample `frag_bands` for prenatal CNV reference branches.
- [`cohort.py:estimate_superblock_size_factors`](./cohort.py) & [`mask_low_coverage_per_contig`](./cohort.py): Compute DESeq2 median-of-ratios library size factors over 1 Mb super-blocks and flag usable bins per contig exceeding 5% of positive median contig coverage.
- [`sex.py:SexInference`](./sex.py), [`SexLabel`](./sex.py), [`infer_germline_sex_2d`](./sex.py), & [`infer_sample_sexes`](./sex.py): Classify germline `XX` vs. `XY` cohort samples via 2D non-PAR dosage separation.
- [`covariates.py:validate_min_mappability`](./covariates.py), [`mask_covariates`](./covariates.py), [`fit_poisson_covariate_trends`](./covariates.py), & [`covariate_features`](./covariates.py): Mask low-mappability/high-N bins, remove per-sample multiplicative GC, mappability, and fragmentomic bias via zero-inclusive `PoissonRegressor` regression on 200 bp GC quantiles, and standardize active covariates.
- [`normalization.py:BranchModel`](./normalization.py), [`extract_reference_branch_arrays`](./normalization.py), [`align_sample_grid_indices`](./normalization.py), [`normalize_sample_branch`](./normalization.py), & [`normalize_branch`](./normalization.py): Project sample coverage onto reference PCA loadings via `HuberRegressor`, evaluate within-sample Prototype Graph baselines, and synthesize depth-decomposed Empirical Bayes precision weights.
- [`fragmentomics.py:extract_bin_fragmentomic_covariates`](./fragmentomics.py), [`extract_sample_fragmentomics`](./fragmentomics.py), [`is_paired_sample_metadata`](./fragmentomics.py), & [`resolve_fragmentomics_compatibility`](./fragmentomics.py): Compute per-bin fragmentomic metrics, decode `FragmentomicsMetadata`, and enforce cohort/sample pairing compatibility.
- [`fragmentomics.py:fit_cohort_fragmentomics_model`](./fragmentomics.py) & [`apply_prenatal_fisher_weighting`](./fragmentomics.py): Deconvolve cohort fetal/maternal length densities across paired-end samples, evaluate sample 24-band Fisher-weighted counts, and fit the 36-feature logit-link `FF_size` ensemble.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py) & [`predict_prenatal`](./prenatal.py): Fit 5 kb `chrY` closed-form Poisson MSY `FFY`, coarsened 100 kb `FFA`, forward $`R_X`$, and fragmentomics models into `prenatal_autosomal` and `prenatal_ffy` branches, and evaluate fetal sex, `FFY`/`FFA`/`FFX`/`FF_size`, 4D erfc ray-marginalized SCA calls, and Student-$`t_4`$ IRLS fused `active_ff`.
- [`prenatal_origin.py:BandPrefixSums`](./prenatal_origin.py), [`classify_segment_origin_5state`](./prenatal_origin.py), [`annotate_prenatal_origin_and_lod`](./prenatal_origin.py), & [`annotate_sample_prenatal_segments`](./prenatal_origin.py): Query masked-bin-excluded $`O(\log U)`$ 24-band prefix sums, evaluate stoichiometric discordance (`SCA_DISCORDANT`, `SUB_FETAL_RATIO`, `SUPER_FETAL_RATIO`), and classify biological origin via the 5-state self-referenced coupled depth + Dirichlet-Multinomial engine (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$).

## Invariants

- [`binning.py:rebin_counts`](./binning.py): Source and target partitions must share identical positive contig lengths and strictly increasing non-overlapping intervals; output counts are non-negative and conserve total read mass.
- [`cohort.py:validate_cohort`](./cohort.py): Rejects mixed paired-end and single-end cohorts (`ReferenceError`) as well as mismatched contig layouts or bin coordinates.
- [`covariates.py:fit_poisson_covariate_trends`](./covariates.py): Operates on usable bins; multiplicative per-sample trends are clamped to $`[0.20, 5.0]`$.
- [`prenatal.py:fit_prenatal_models`](./prenatal.py): Requires at least one sample table, gracefully handling single-sex or unlabeled-FF cohorts via stoichiometric fallbacks.

## Side Effects

- None

## Verification

- Run: `pixi run test`
