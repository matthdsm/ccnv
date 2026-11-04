# `src`

## Overview

The `src` directory hosts the top-level Python namespace and implementation packages for the `copynumber-cnv` toolkit. Source modules under this root adhere to strict 0-based half-open coordinate contracts, versioned Apache Arrow/Parquet artifact specifications, and immutable bioinformatic pipelines.

## Subdirectories & Boundaries

- `ccnv`: [ccnv/README.md](./ccnv/README.md) - Core bioinformatics pipeline implementation encompassing alignment streaming, robust Huber covariate bias correction, principal component analysis, off-contig nearest-bin reference modeling, weighted Ruptures change-point segmentation, and genomic reporting.
