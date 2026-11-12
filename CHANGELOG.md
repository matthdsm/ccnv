# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> [!NOTE]
> Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

## [Unreleased]

### Added

- Compiled Rust I/O kernel (`crates/ccnv-core`, `ccnv._core`) wrapping `noodles` (`bam`, `cram`, `fasta`, `bgzf`, `tabix`, `csi`) and `bigtools` (BigWig) for high-performance multi-threaded tile streaming and single-pass BGZF/tabix/CSI writing without runtime dependencies on `pysam` or `pybigwig`.
- Apache Arrow Schema `2.0.0` (`SCHEMA_VERSION = "2.0.0"`) storing 7 coordinate-free per-bin sample columns (`count`, `paired_count`, `frag_len_sum`, `short_frag_count`, `endpoint_count`, `frag_gc_sum`, and 24 fixed fragment-length bands `frag_bands: list<uint32>[24]`) for compact $`O(B)`$ memory footprint.
- Compact Prototype Graph ($`M \le 2{,}048`$) constructed via `MiniBatchKMeans` with $`k=4`$ soft-weighted nearest off-contig prototypes, eliminating $`O(U^2)`$ pairwise distance calculations.
- Depth-decomposed biological overdispersion (`bio_dispersion`) separating Poisson shot noise from technical dispersion using Smyth Empirical Bayes shrinkage (`squeeze_var`).
- Calibrated `ruptures.Pelt` change-point segmentation with modified BIC (mBIC) penalty $`\beta_c = \gamma \cdot 2 \hat{\kappa}_c \ln(\max(U_{\text{block}}, 3))`$.
- Multi-scale dyadic BLUE segment Z-scores ($`Z_{\text{BLUE}}`$) with spatial variance inflation $`\hat{\kappa}_c(L)`$ across dyadic block scales $`L \in \lbrace 1, 4, 16, 64, 256, 1024 \rbrace`$.
- Discrete integer copy-number Bayes factors ($`\ln \mathrm{BF}_{CN \ne CN_0} \gt 2.0`$) over diploid ($`CN \in \lbrace 0, 1, 2, 3, 4+\rbrace`$) and haploid sex-chromosome ($`CN \in \lbrace 0, 1, 2, 3+\rbrace`$) states using segment Negative-Binomial/Poisson count log-likelihood ratios (with Gaussian log-ratio fallback when counts are absent), detecting homozygous deletions ($`CN=0`$) down to $`0.1\times`$ sequencing depth.
- Closed-form Poisson MLE for MSY fetal fraction (`FFY`) with analytical standard error propagation.
- Forward-calibrated $`R_X`$ baseline regression across euploid `XX` ($`f=0`$) and `XY` ($`f=\hat{f}_{\text{FFY}}`$) control cohorts.
- Reference-side 24-band Fisher read weighting ($`w^{\ast}_{0.08}`$) pre-applied during reference construction, eliminating test-time baseline distortions.
- 4D closed-form erfc Gaussian mixture ray-marginalized sex chromosome aneuploidy (SCA) classifier with `chrX` fragment-origin likelihood gating.
- Precision-weighted Student-$`t_4`$ IRLS multi-modal active fetal fraction fusion (`active_ff`).
- 5-state self-referenced coupled depth + Dirichlet-Multinomial origin engine (`FETAL_CONSTITUTIONAL`, `PLACENTAL_MOSAIC`, `MATERNAL_UNINHERITED`, `MATERNAL_AND_FETAL_INHERITED`, `MATERNAL_MOSAIC`, with `INDETERMINATE` when max posterior $`\lt 0.80`$).
- Multi-architecture `abi3` release wheel matrix in GitHub Actions across `ubuntu-latest` (`x86_64`), `ubuntu-24.04-arm` (`aarch64`), and `macos-latest` (`arm64`), plus `sdist`.
- Slim production Dockerfile copying compiled Rust crates and stripping build artifacts (`rustlib`, `rustc`, `cargo`) before production image packaging.

### Changed

- Replaced Python/pysam alignment streaming with compiled Rust multi-threaded tile scanner.
- Replaced $`k=300`$ `KDTree` all-bin nearest neighbor search with compact Prototype Graph ($`M \le 2{,}048`$, $`k=4`$).
- Replaced greedy `Binseg` default with calibrated `ruptures.Pelt` + mBIC.
- Replaced single-sample runtime Fisher weight scaling with reference-side 24-band weighting ($`w^{\ast}_{0.08}`$) applied consistently across prenatal cohort CNV reference branches and sample prediction.
- Updated `--pca-components` (`ReferenceConfig.pca_components`) to default to `None` (dynamic Gavish-Donoho SVHT rank selection up to `60`) while treating explicit integer values as hard upper caps.
- Canonicalized in-memory sample representation on `SampleBuffers`, unified 17-column reference branch assembly on `BranchFitArrays` and `assemble_reference_table`, and consolidated single-pass FASTA GC and 200 bp GC quantile scanning in `load_gc_track`.
- Updated prenatal sex chromosome aneuploidy classification from 2D Mahalanobis $`\chi^2`$ gate to 4D erfc ray marginalization.
- Upgraded sample and reference storage contract from `1.0.0` to `2.0.0`.

### Removed

- Removed runtime dependencies on `pysam`, `pybigwig`, and `samtools` from production environment.
- Removed deprecated `--nearest-bins` CLI flag and positional `ccnv reference` sample arguments in favor of `--samplesheet` and `--fasta`.
- Removed legacy `align_reference_to_weighted_sample` function and unused v1 kNN, `Binseg`, Stouffer, and B-spline helpers.
- Removed legacy `fragment_lengths: list<uint16>` and `nearest_ref_indices: list<int32>` schema columns.
