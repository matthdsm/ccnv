# `src`

## Overview

The `src` directory hosts the top-level Python namespace and implementation packages for the `copynumber-cnv` toolkit. Source modules under this root enforce 0-based half-open coordinate contracts, versioned Apache Arrow/Parquet (`1.0.0`) storage specifications, and deterministic bioinformatic pipelines.

## Subdirectories & Boundaries

- `ccnv`: [ccnv/README.md](./ccnv/README.md) - Core bioinformatics pipeline implementation covering BAM/CRAM streaming and cell-free DNA fragmentomics extraction, DESeq2 median-of-ratios depth normalization, joint B-spline Huber covariate bias correction, Gavish-Donoho SVHT PCA, signal-subspace `KDTree` reference modeling, Smyth Empirical Bayes variance moderation, reference QC auditing, germline and prenatal NIPT inference, weighted Ruptures change-point segmentation, and tabix-indexed/HTML reporting.
