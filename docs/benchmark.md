# Benchmarks & Concordance with WisecondorX

This document summarizes the numerical concordance, storage footprint, and algorithmic differences between `ccnv` and WisecondorX across extraction, reference construction, and prediction on shallow whole-genome sequencing (sWGS) data.

## Architectural & Algorithmic Summary

| Dimension | WisecondorX | `ccnv` |
| --- | --- | --- |
| **Storage Format** | Unindexed NumPy `.npz` archives | Versioned (`1.0.0`) Zstandard/Delta-packed Apache Arrow Parquet (`.ccnv`) |
| **Coordinates** | Implicit bin indices in `.npz`; 1-based BED outputs | Explicit `contig`, `start`, `end` columns; 0-based half-open `[start, end)` BED and 1-based VCFv4.2 |
| **Genome Layout** | Hardcoded 24-chromosome human layout (`1`–`22`, `23=X`, `24=Y`) | Organism-agnostic; contig names, order, and lengths derived from alignment/FASTA headers |
| **Bin Resolution** | Sample and reference bin sizes must match identically | Dynamic mass-conserving in-memory rebinning during `reference` and `predict` |
| **Duplicate Filtering** | Single-read adjacent coordinate streaming check | Standard SAM duplicate flag (`0x400`) |
| **Segmentation Engine** | Unweighted R `DNAcopy` (CBS) via `rpy2` (`alpha`) | Pure-Python weighted `ruptures` Binary Segmentation (`Binseg`) with precision weights $w = 1 / (\sigma_{\text{null}}^2 + \epsilon)$ |
| **Deliverables** | Plain-text BED and PNG plots | Bgzipped/tabix-indexed BED (`.bed.gz`), VCFv4.2 (`.vcf.gz`), and interactive Plotly HTML (`_plot.html`) |

## 1. Extraction (`extract`) Benchmark

Evaluated on a real-world NIPT sample binned at 5 kb across 24 chromosomes (`chr1`–`chr22`, `chrX`, `chrY`), totaling 617,665 bins and 66,629,087 raw reads.

### Read-Count & Bin-Level Concordance

- **Total reads examined**: 66,629,087 in both tools.
- **Total reads retained**: 56,781,413 in `ccnv` vs. 56,649,323 in WisecondorX (+132,090 reads; +0.23%).
- **Correlation across all 617,665 bins**: Pearson $r = 0.9779$ ($R^2 = 0.9562$); Spearman $\rho = 0.9995$.
- **Error distribution**: Median absolute difference of 0.0 reads/bin (mean absolute difference: 0.522 reads/bin).
- **Bin agreement**:
  - Exact match: 66.69% (411,926 / 617,665 bins)
  - Within $\pm 1$ read: 91.70% (566,413 / 617,665 bins)
  - Within $\pm 2$ reads: 98.66% (609,418 / 617,665 bins)
  - Within $\pm 5$ reads: 99.96% (617,438 / 617,665 bins)

### Cause of Count Differences

Only 56 of 617,665 bins (< 0.01%) differ by more than 50 reads, all located in pericentromeric heterochromatin and satellite repeats:

1. **Duplicate detection**: `ccnv` uses the SAM duplicate bit (`0x400`). WisecondorX compares only the immediately preceding read's start and mate start coordinates; interleaved pairs in coordinate-sorted BAM/CRAM streams bypass its check in repeat-dense bins (e.g., 16,565 duplicate reads retained in a single `chr3` centromeric bin) or cause false drops when a primary read follows an identically positioned duplicate.
2. **Alignment flags**: `ccnv` explicitly filters unmapped, secondary (`0x100`), and supplementary (`0x800`) alignments prior to duplicate and proper-pair checks, whereas WisecondorX relies on `MAPQ < 1`.

### File Size Analysis

- **Count column**: `ccnv` compresses the 617,665-row `count` column with Zstandard to 637 KB, smaller than the 723 KB `.npz` array.
- **Explicit coordinates**: The total `.ccnv` sample file is 3.28 MB because it stores explicit 64-bit `start` (1.32 MB) and `end` (1.32 MB) coordinates and QC metadata rather than implicit array offsets.

## 2. Reference Construction (`reference`) Benchmark

Evaluated on a 100 kb GRCh38 reference model built from a 299-sample NIPT control cohort.

### Numeric Concordance (26,800 Mutually Unmasked Autosomal Bins)

- **Bin retention**: `ccnv` retains 100% of WisecondorX's unmasked autosomal bins (26,800 / 26,800) plus 632 additional informative bins (27,432 total unmasked) that WisecondorX discarded via post-PCA Euclidean distance pruning.
- **Mean coverage profile**: Pearson $r = 0.994504$; Spearman $\rho = 0.999834$ (mean absolute difference $4.39 \times 10^{-8}$).
- **PCA loadings**: Component 1 $r = 0.9761$; Component 2 $r = 0.8899$; Component 3 $|r| = 0.8748$; Component 4 $r = 0.8403$; Component 5 $r = 0.9568$.
- **Null noise estimation (`null_std`)**: Pearson $r = 0.907506$ against WisecondorX empirical null standard deviations.

### Storage Efficiency (9.2x Smaller Reference File)

The `ccnv` reference `.ccnv` file is **13.3 MB** compared to **123.2 MB** for WisecondorX:

1. **Scalar mean distance vs. full distance matrix (-56.3 MB)**: WisecondorX serializes 300 `float64` distances per bin (`distances.npy`: 56.3 MB) that are unused during prediction. `ccnv` stores `mean_distance` (`float32`: 93 KB) for QC.
2. **Sufficient null statistics vs. empirical matrix (-42.2 MB)**: WisecondorX stores 100 empirical null ratios per bin across autosomal and female branches (42.2 MB). `ccnv` stores `null_mean` (222 KB) and `null_std` (215 KB).
3. **Disjoint branch rows**: WisecondorX duplicates all 28,760 autosomal bins inside its `.F` arrays; `ccnv` stores 28,760 `autosomal` rows and 2,134 `female_sex` rows in a single table with `DELTA_BINARY_PACKED` neighbor indices.

## 3. Prediction & Segmentation (`predict`) Benchmark

Evaluated on a real-world NIPT sample normalized against the 100 kb 299-sample GRCh38 reference models, as well as published real trisomy 21 and synthetic fixtures.

### Bin-Level & Segment-Level Concordance

- **Unmasked bin recovery**: `ccnv` recovers 100% of WisecondorX's 27,129 unmasked bins plus 695 additional bins (632 autosomal + 63 sex-contig bins; 27,824 total).
- **$\log_2$ ratio concordance**: Pearson $r = 0.986851$; Spearman $\rho = 0.985708$; median absolute error $0.004394$ (mean $0.006006$).
- **Z-score concordance**: Pearson $r = 0.981460$; Spearman $\rho = 0.984927$; median absolute error $0.343467$ (mean $0.357725$).
- **Segment & call agreement**:
  - On the negative NIPT sample, `ccnv` produced 44 autosomal/`chrX` segments (plus 4 `chrY` segments) vs. 45 in WisecondorX, with 0 calls in both tools (100% negative concordance).
  - On the real chromosome 21 gain benchmark (`tests/fixtures/benchmark/real/`), 50 of 51 `ccnv` segments match within one 100 kb bin and the `chr21:13100000-46700000` gain is recovered with 100% recall.
  - On the synthetic benchmark (`tests/fixtures/benchmark/synthetic/`), segment means agree within $10^{-12}$ and calls match identically.

## 4. Automated Verification

Run the benchmark and contract test suite locally with:

```console
pixi run test
pixi run lint
pixi run hooks
```

- `tests/test_baseline.py` and `tests/test_segment.py` verify synthetic and real-world segmentation and calling concordance against `tests/fixtures/benchmark/`.
- `tests/test_pipeline.py` runs end-to-end BAM and CRAM conversion, reference building, and prediction across arbitrary contig layouts and partial final bins.
