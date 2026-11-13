# `src`

## Overview

The `src` directory hosts the top-level Python namespace and implementation packages for the `ccnv` toolkit. Source modules under this root enforce 0-based half-open coordinate contracts, versioned Apache Arrow/Parquet (`2.0.0`) storage specifications, and deterministic bioinformatic pipelines.

## Subdirectories & Boundaries

- `ccnv`: [ccnv/README.md](./ccnv/README.md) - Core bioinformatics pipeline implementation covering compiled Rust tile streaming (`ccnv._core`) and cell-free DNA 24-band fragmentomics extraction, 1 Mb super-block DESeq2 median-of-ratios depth normalization, zero-inclusive `PoissonRegressor` covariate bias correction, whitened Gavish-Donoho SVHT PCA, Prototype Graph ($`M \le 2{,}048`$) reference modeling, Smyth Empirical Bayes variance moderation, reference QC auditing, germline and prenatal NIPT inference, calibrated Pelt change-point segmentation with mBIC, dyadic BLUE scoring, discrete integer copy-number Bayes factors, and single-pass tabix-indexed/HTML reporting.
