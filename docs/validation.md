# Validation

Validation targets scientific agreement and explicit contracts, not legacy file
bytes. Run it with:

```console
pixi run test
pixi run lint
pixi run hooks
```

## Evidence

- `tests/test_pipeline.py` runs indexed synthetic BAM and CRAM controls through
  conversion and reference construction, then predicts a BAM containing a known
  `chr10` gain and `linkage-X` loss. It checks BED bin/segment/call artifacts
  (genome contig names, 0-based half-open coordinates), a partial final bin,
  arbitrary contig names/order, and segment bounds.
- `tests/test_baseline.py` compares numeric segment means, breakpoints, and calls
  with immutable WisecondorX 1.3.2 observations in
  `tests/fixtures/upstream/`. The synthetic means agree to `1e-12` and calls are
  exactly concordant. The published real gain is recovered with both boundaries
  within one 100 kb bin.
- Contract and unit tests cover incompatible/missing contigs, `chr10`, 0-based
  half-open BED output, partial contig ends, arbitrary X/Y-like names, all-zero
  weights, and deterministic cohort/sex modeling.
- Conversion closes its alignment handle on validation failure. Wheel smoke
  testing verifies that installation and import do not invoke R.

## Accepted differences from 1.3.2

- **Storage:** Versioned Parquet replaces NPZ; byte compatibility is
  intentionally absent.
- **Coordinates:** All intervals are 0-based half-open. Published upstream
  BED-like fixtures are converted from 1-based inclusive coordinates in tests.
- **Segmentation:** Weighted Binseg with an absolute weighted-SSE penalty
  replaces DNAcopy. The penalty is not DNAcopy `alpha`. Synthetic means/calls
  agree; breakpoints may differ by one bin.
- **Published real sample:** 50 of 51 rewrite segments match an upstream segment
  within one 100 kb bin. The known chromosome 21 gain is recovered. At a direct
  absolute-ratio threshold, call recall is 1.0 and precision is at least 0.5
  because upstream additionally used its own z-score/null model.
- **Sex chromosomes:** Explicit branch metadata and validated labels/inference
  replace suffix conventions and the legacy two-component Y-fraction
  assumption.
- **Genome layout:** Header contig names and true lengths are authoritative;
  numeric IDs and a 24-chromosome assumption are not supported.

These differences are migration boundaries, not promises that future versions
will reproduce DNAcopy breakpoint placement.

## Previously observed upstream failures

- Omitted `chr10`: the end-to-end layout explicitly contains and calls `chr10`.
- BED ambiguity: output tests assert start `0` and an unchanged exclusive end.
- Segments beyond contig ends: schema and segmentation reject invalid bins and
  clip output to declared lengths.
- X/Y parsing: contig names are opaque strings (`X`, `chrY`, `linkage-X`, and
  `plasmid-Y` are covered).
- All-zero weights: segmentation returns no segments without crashing.
- Missing contigs: prediction fails before normalization with the contig name.
- File-handle cleanup: conversion closes BAM/CRAM handles through `finally`.
- Pip installs without R: R, rpy2, r-base, and DNAcopy are absent from package
  dependencies; Python-native `ruptures` performs segmentation.

The fixture set is compact and does not establish clinical sensitivity,
specificity, mosaicism limits, or cross-platform floating-point identity.
