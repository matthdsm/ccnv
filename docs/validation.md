# Validation

Validation targets scientific agreement and explicit contracts, not legacy file bytes. Run it with:

```console
pixi run test
pixi run lint
pixi run hooks
```

## Evidence

- `tests/test_pipeline.py` runs indexed synthetic BAM and CRAM controls through conversion and reference construction, then predicts a BAM containing a known `chr10` gain and `linkage-X` loss. It checks BED bin/segment/call artifacts (genome contig names, 0-based half-open coordinates), a partial final bin, arbitrary contig names/order, and segment bounds.
- `tests/test_baseline.py` compares numeric segment means, breakpoints, and calls with immutable WisecondorX 1.3.2 observations in `tests/fixtures/upstream/`. The synthetic means agree to `1e-12` and calls are exactly concordant. The published real gain is recovered with both boundaries within one 100 kb bin.
- Contract and unit tests cover incompatible/missing contigs, `chr10`, 0-based half-open BED output, partial contig ends, arbitrary X/Y-like names, all-zero weights, and deterministic cohort/sex modeling.
- Conversion closes its alignment handle on validation failure. Wheel smoke testing verifies that installation and import do not invoke R.

## Accepted differences from 1.3.2

- **Storage:** Versioned Parquet replaces NPZ; byte compatibility is intentionally absent.
- **Coordinates:** All intervals are 0-based half-open. Published upstream BED-like fixtures are converted from 1-based inclusive coordinates in tests.
- **Segmentation:** Weighted Binseg with an absolute weighted-SSE penalty replaces DNAcopy. The penalty is not DNAcopy `alpha`. Synthetic means/calls agree; breakpoints may differ by one bin.
- **Published real sample:** 50 of 51 rewrite segments match an upstream segment within one 100 kb bin. The known chromosome 21 gain is recovered. At a direct absolute-ratio threshold, call recall is 1.0 and precision is at least 0.5 because upstream additionally used its own z-score/null model.
- **Sex chromosomes:** Explicit branch metadata and validated labels/inference replace suffix conventions and the legacy two-component Y-fraction assumption.
- **Genome layout:** Header contig names and true lengths are authoritative; numeric IDs and a 24-chromosome assumption are not supported.

These differences are migration boundaries, not promises that future versions will reproduce DNAcopy breakpoint placement.

## Previously observed upstream failures

- Omitted `chr10`: the end-to-end layout explicitly contains and calls `chr10`.
- BED ambiguity: output tests assert start `0` and an unchanged exclusive end.
- Segments beyond contig ends: schema and segmentation reject invalid bins and clip output to declared lengths.
- X/Y parsing: contig names are opaque strings (`X`, `chrY`, `linkage-X`, and `plasmid-Y` are covered).
- All-zero weights: segmentation returns no segments without crashing.
- Missing contigs: prediction fails before normalization with the contig name.
- File-handle cleanup: conversion closes BAM/CRAM handles through `finally`.
- Pip installs without R: R, rpy2, r-base, and DNAcopy are absent from package dependencies; Python-native `ruptures` performs segmentation.

The fixture set is compact and does not establish clinical sensitivity, specificity, mosaicism limits, or cross-platform floating-point identity.

## Conversion concordance and storage comparison against WisecondorX

Validation of the `convert` stage was conducted on a real-world NIPT sample binned at 5 kb across 24 chromosomes (1–22, X, Y), comprising 617,665 total bins and 66,629,087 raw reads.

### Read count and bin-level concordance

- **Total reads examined:** Exactly identical between both tools (66,629,087 reads).
- **Total reads kept:** 56,781,413 reads in `ccnv` versus 56,649,323 reads in WisecondorX (+132,090 reads, a 0.23% delta).
- **Correlation across all 617,665 bins:** Pearson $r = 0.9779$ ($R^2 = 0.9562$); Spearman rank correlation $\rho = 0.9995$.
- **Error distribution:** Median absolute difference is 0.0 reads/bin (mean absolute error: 0.522 reads/bin).
- **Bin agreement thresholds:**
  - Exact match: 66.69% of bins (411,926 / 617,665).
  - Within $\pm 1$ read: 91.70% of bins (566,413 / 617,665).
  - Within $\pm 2$ reads: 98.66% of bins (609,418 / 617,665).
  - Within $\pm 5$ reads: 99.96% of bins (617,438 / 617,665).

### Root cause of count discrepancies

Only 56 of 617,665 bins (< 0.01%) exhibit an absolute difference exceeding 50 reads. All high-difference bins locate to pericentromeric heterochromatin and satellite repeat regions (e.g., centromeres of chromosomes 1, 3, 16, 17, and 20).

1. **Duplicate filtering mechanism:** `ccnv` respects the standard SAM specification duplicate bit (`read.is_duplicate` / flag 0x400) populated by upstream duplicate markers. WisecondorX employs an ad-hoc single-read streaming check (`larp == read_start and larp2 == next_reference_start`). Because reads in coordinate-sorted alignments can interleave across strands and pairs, WisecondorX fails to identify duplicate pairs that are not strictly adjacent in the stream. In repeat-dense regions, thousands of duplicate reads bypass the WisecondorX filter (for example, missing 16,565 duplicate reads in a single chromosome 3 centromeric bin). Conversely, WisecondorX can mistakenly discard primary non-duplicate reads that stream immediately after an identically positioned duplicate.
2. **Filter precedence:** `ccnv` filters duplicates before validating proper pair status (`filter_duplicate`: 6,546,411; `filter_improper_pair`: 302,029). WisecondorX evaluates proper pair status prior to duplicate checking (`pair_fail`: 409,495; `rmdup`: 6,440,137), causing improper-pair duplicates to be classified as pair failures rather than PCR duplicates.
3. **Unmapped and secondary alignments:** `ccnv` explicitly drops unmapped reads (`filter_unmapped`: 64,265), secondary reads, and supplementary alignments. WisecondorX streams alignment records without inspecting unmapped or secondary flags directly, relying solely on MAPQ < 1.

### Storage architecture and file size analysis

The original WisecondorX `.npz` file occupies ~724 KB, whereas the corresponding `ccnv` `.ccnv` Parquet table occupies ~3.28 MB.

1. **Implicit versus explicit coordinates:** The `.npz` file stores no genomic coordinates. It is a compressed zip archive of 1D `int32` count arrays per chromosome where bin coordinates are completely implicit (`bin_index * binsize`). In contrast, `ccnv` produces a self-describing, organism-agnostic Arrow Parquet table storing explicit `contig` (string), `start` (int64), and `end` (int64) columns adhering to 0-based half-open intervals, alongside deterministic JSON metadata (QC counters, contig lengths, filter parameters).
2. **Column-level size breakdown:** The `count` column in `ccnv` Parquet is compressed with Zstandard to 637 KB—which is smaller than the count array in the `.npz` file (723 KB). The remaining 2.63 MB (80.5% of the file) is consumed by the explicit 64-bit integer coordinate columns (`start` at 1.32 MB, `end` at 1.32 MB). Parquet default dictionary encoding falls back to uncompressed plain integer encodings when unique coordinate counts exceed dictionary thresholds.

## Reference creation concordance and storage comparison against WisecondorX

Validation of the `reference` stage was conducted by comparing a 100 kb reference model built by `ccnv reference` against the reference model produced by `wisecondorx newref` using the same control cohort of 299 NIPT sequencing samples across GRCh38.

### Step-by-step algorithmic comparison

1. **Cohort ingestion and layout validation:** `wisecondorx` hardcodes a 24-chromosome human layout (1–22, 23=X, 24=Y) and assumes implicit bin coordinates from array lengths. `ccnv` reads versioned sample Parquet tables, validates contig naming and lengths against the authoritative alignment headers, and supports multithreaded loading and in-memory rebinning across arbitrary reference layouts.
2. **Depth normalization:** `wisecondorx` normalizes each sample across all 24 chromosomes globally. `ccnv` normalizes samples per branch across the active contigs of that model branch (`normalize_depth`).
3. **Sex inference and branch construction:** `wisecondorx` fits a 2-component Gaussian Mixture Model (GMM) to 1D chromosome Y read fractions. `ccnv` evaluates coverage on the sex-signal contig (`chrY`) relative to total depth, placing an inference cutoff using cluster separation relative to Median Absolute Deviation (MAD), or accepting an explicit `--sex-cutoff`. In this 299-sample NIPT panel, both tools classified all samples below the male threshold, producing an `autosomal` branch and a `female_sex` branch without a male reference branch.
4. **Low-coverage masking and bin pruning:** `wisecondorx` masks bins with zero counts or coverage below 5% of median positive coverage, followed by an aggressive post-PCA Euclidean distance filter (`_robust_cutoff`). `ccnv` applies coverage thresholding (`mask_low_coverage`) while preserving biological variance without post-PCA outlier dropping. Across 28,760 autosomal bins, `ccnv` keeps 27,432 unmasked bins (1,328 masked) while `wisecondorx` keeps 26,800 bins (1,960 masked). **100% of WisecondorX's usable bins (26,800 / 26,800) are retained by `ccnv`**, with `ccnv` recovering 632 additional informative bins previously discarded by PCA distance pruning.
5. **PCA bias modeling:** Both pipelines perform 5-component PCA on centered sample-by-bin coverage matrices and reconstruct systemic batch biases to obtain corrected depth ratios (`samples / reconstructed`).
6. **Nearest off-contig bin selection:** `wisecondorx` computes squared Euclidean distances to candidate bins on different chromosomes using a looping search with `bisect`. `ccnv` accelerates selection with multithreaded BLAS matrix multiplication bounded by memory batches, masks the query chromosome contiguously (`row[c_start:c_end] = inf`), and partitions top candidates with `np.argpartition`.
7. **Null distribution modeling:** `wisecondorx` randomly samples up to 100 cohort samples and saves the entire $(26800, 100)$ matrix of empirical log2 ratios into the reference file. `ccnv` computes the null log2 ratio distribution across the cohort in parallel chunks, storing only the necessary summary statistics: `null_mean` (double) and `null_std` (double).
8. **Storage serialization:** `wisecondorx` stores unindexed `.npy` arrays inside a compressed zip archive, duplicating autosomes inside the female reference branch (`.F`). `ccnv` serializes a single, unified Parquet table partitioned by `branch` (`autosomal` with 28,760 bins and `female_sex` with 2,134 bins) compressed with Zstandard and Delta Binary Packing.

### Numeric concordance across mutually unmasked bins (26,800 bins)

- **Mean coverage profile:** Pearson $r = 0.994504$; Spearman rank correlation $\rho = 0.999834$. Mean absolute difference across bins is $4.39 \times 10^{-8}$.
- **PCA component loadings:**
  - Component 1: Pearson $r = 0.976105$
  - Component 2: Pearson $r = 0.889925$
  - Component 3: Pearson $r = -0.874788$ ($|r| = 0.874788$, reflecting expected SVD sign invariance)
  - Component 4: Pearson $r = 0.840262$
  - Component 5: Pearson $r = 0.956770$
- **Null model noise estimation:** Bin standard deviations (`null_std`), which dictate weighted Ruptures Binseg penalty weights $w = 1 / (\sigma^2 + \epsilon)$, correlate with WisecondorX empirical null standard deviations at **Pearson $r = 0.907506$**.

### Storage footprint and architectural efficiency analysis

The `ccnv` reference file is **9.2x smaller** than the WisecondorX reference file (13.3 MB versus 123.2 MB) despite storing explicit genomic coordinates, contig names, and rich provenance metadata:

1. **Elimination of pairwise distance matrix (saves ~56.3 MB):** WisecondorX stores the raw matrix of 300 float64 distances for each bin (`distances.npy`: 56.3 MB compressed). During prediction, individual neighbor distances are unused; only neighbor identities are needed to look up sample coverage. `ccnv` stores a single scalar `mean_distance` (float32: 93 KB) for quality control, reducing distance storage by over 600x.
2. **Sufficient statistics for null modeling (saves ~42.2 MB):** WisecondorX serializes 100 empirical float64 null ratios per bin across both autosomal and female branches (`null_ratios.npy` at 20.5 MB and `null_ratios.F.npy` at 21.7 MB). `ccnv` stores `null_mean` (222 KB) and `null_std` (215 KB), reducing null model storage by over 96x without loss of segmentation weighting accuracy.
3. **Disjoint relational branches (eliminates redundant autosomes):** WisecondorX duplicates all 28,760 autosomes in its female `.F` reference arrays, populating autosome entries with dummy zero indices. `ccnv` stores a clean disjoint schema: the `autosomal` branch contains 28,760 bins, and `female_sex` contains only the 2,134 sex chromosome bins (`chrX` and `chrY`).
4. **Columnar Delta Binary Packing:** `ccnv` encodes sorted integer lists (`nearest_ref_indices.list.element`) and coordinates (`start`, `end`) using Parquet `DELTA_BINARY_PACKED` encoding, compressing 300 neighbor indices per bin into 11.3 MB with fast random access.

## Prediction concordance and segmentation comparison against WisecondorX

Validation of the `predict` stage was conducted by running `ccnv predict` and `wisecondorx predict` on a real-world NIPT sequencing sample against matching 100 kb reference models trained on the same 299-sample control cohort across GRCh38.

### Step-by-step algorithmic comparison

1. **Input ingestion and dynamic rebinning:** `wisecondorx` requires the sample file to match the reference bin size identically, aborting if dimensions differ. `ccnv` inspects alignment header contig lengths and dynamically aggregates sample counts from native resolution (5 kb) to reference resolution (100 kb) in memory before projection, leaving reference and input Parquet tables untouched.
2. **Depth scaling and sex assignment:** `wisecondorx` scales sample depth across autosomes and classifies sample sex via reference metadata thresholds (`--yfrac`), skipping chromosome Y entirely for female cases. `ccnv` normalizes depth per model branch (`autosomal` and `female_sex`) and evaluates sex chromosomes against reference sex models, retaining unmasked pseudoautosomal and non-silent Y bins.
3. **Between-sample PCA normalization:** Both pipelines project centered sample depth onto the 5-component PCA basis, reconstruct systemic bias profiles, and compute bias-corrected depth ratios (`sample / reconstructed`).
4. **Within-sample reference normalization:** `wisecondorx` computes log2 depth ratios against the median of 300 nearest reference bins using an index shifted by chromosome excision. `ccnv` directly looks up the nearest reference bins in the branch's unmasked index space and evaluates median reference depth: $\log_2(\text{sample}[\text{bin}] / \text{median}(\text{reference\_bins}))$.
5. **Z-score and precision weight calculation:** `wisecondorx` computes bin z-scores from the 100 empirical null ratios stored in the reference archive, without computing bin weights for segmentation. `ccnv` evaluates z-scores parametrically from `null_mean` and `null_std`, and derives precision weights $w = 1 / (\sigma_{\text{null}}^2 + \epsilon)$ to prioritize high-confidence bins during segmentation.
6. **Segmentation engine:** `wisecondorx` delegates segmentation to the R package `DNAcopy` (Circular Binary Segmentation / CBS) via `rpy2`, running permutation tests with equal weights across all bins. `ccnv` uses Python-native `ruptures` Weighted Binary Segmentation (`Binseg`), penalizing weighted sum-of-squared errors without R dependencies or external subprocesses.
7. **Aberration and CNV calling:** `wisecondorx` filters segments with absolute z-scores exceeding 5.0 into an aberrations BED file. `ccnv` classifies segments exceeding the call threshold (`--call-zscore 5.0`) into gains and losses. **Both tools produced exactly 0 calls (100% negative concordance)** on this sample.
8. **Artifact reporting:** `wisecondorx` outputs 1-based plain text BED tables and PNG plots. `ccnv` generates standard 0-based half-open BED tables, typed columnar Parquet tables (`_bins.ccnv`, `_segments.ccnv`), and resolution-independent vectorized SVG plots (`_plot.svg`).

### Bin-level numeric concordance across 27,129 mutually unmasked bins

- **Log2 ratio agreement:** Pearson $r = 0.986851$; Spearman rank correlation $\rho = 0.985708$. Mean absolute error across all bins is $0.006006$ log2 ratio units (median absolute error: $0.004394$).
- **Z-score agreement:** Pearson $r = 0.981460$; Spearman rank correlation $\rho = 0.984927$. Mean absolute error is $0.357725$ (median absolute error: $0.343467$).
- **Bin coverage and unmasked recovery:** `wisecondorx` produced valid ratios for 27,129 bins. `ccnv` produced valid ratios for 27,824 bins. **100% of WisecondorX unmasked bins (27,129 / 27,129) are recovered by `ccnv`**, with `ccnv` providing 695 additional informative bins (632 autosomal bins retained by improved reference masking plus 63 sex-contig bins).

### Segment-level concordance and breakpoint stability

- **Total segment counts:** `ccnv` produced 48 segments across 24 contigs (1–22, X, Y), including 4 segments on pseudoautosomal/active regions of `chrY`. `wisecondorx` produced 45 segments across 23 contigs (1–22, X), omitting `chrY`. Excluding `chrY`, segment counts are nearly identical: **44 in `ccnv` versus 45 in WisecondorX**.
- **Breakpoint alignment:** Major chromosome arm boundaries and centromeric gap coordinates match across all autosomes and sex contigs (for example, chromosome 3 as a single continuous arm; chromosome 4 centromere gap at 49.2–51.8 Mb; chromosome 10 centromere gap at 39.5–41.7 Mb; chromosome 18 at 15.4–20.9 Mb; and chromosome 19 at 24.4–27.2 Mb).
- **Clinical call concordance:** Both tools produced 0 calls (negative concordance: 100%), confirming that minor segmentation boundary shifts on noisy bins do not introduce false-positive aberration calls.

### Storage footprint and format advantages

1. **Columnar Parquet compression (4.2x smaller):** The `ccnv` bin table (`_bins.ccnv`) is compressed with Zstandard to 732 KB, representing a 4.2x reduction in size compared to the corresponding BED file (3.06 MB) while preserving native data types (floats, booleans, strings) and dictionary-encoded contig metadata.
2. **Deterministic, zero-dependency execution:** Replacing R/rpy2 CBS with Python-native weighted `ruptures` removes all R runtime dependencies, ensuring portable execution across Linux, macOS, and containerized Docker environments without numerical divergence from R package version drift.
3. **Standardized coordinate contracts:** All BED exports from `ccnv` use 0-based half-open coordinates with UCSC-style contig naming (`chr1`–`chr22`, `chrX`, `chrY`), ensuring direct compatibility with IGV, UCSC Genome Browser, and downstream genomic pipelines without manual coordinate shifting.