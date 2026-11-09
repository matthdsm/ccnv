# Pipeline Architecture

`ccnv` executes copy-number variation (CNV) analysis across four stages: [`extract`](#1-extract) (bins alignments into sample Parquet), [`reference`](#2-reference) (builds cohort or flat FASTA reference models), [`ref-qc`](#3-ref-qc) (audits reference quality), and [`predict`](#4-predict) (normalizes coverage, segments with weighted change-point solvers, and calls aberrations).

```mermaid
flowchart LR
    A["BAM / CRAM Alignment"] -->|ccnv extract| B["Sample Parquet (<sample>.ccnv)"]
    C["Control BAMs / CRAMs"] -->|ccnv extract| D["Control Parquets (<ctrl>.ccnv)"]
    D -->|ccnv reference| E["Reference Parquet (<reference>.ccnv)"]
    D -->|ccnv reference| H["Reference QC (<reference>_qc.json)"]
    F["Reference FASTA"] -.->|ccnv reference (flat)| E
    E -->|ccnv ref-qc| H
    B -->|ccnv predict| G["Prediction Artifacts (BED.gz, VCF.gz, HTML)"]
    E -->|ccnv predict| G
```

## Coordinate & Storage Contracts

- **0-Based Half-Open Coordinates**: All internal bins, segments, and BED outputs use `[start, end)` intervals. VCFv4.2 outputs use 1-based coordinates per the VCF specification.
- **Header-Authoritative Layouts**: Contig names, ordering, and exact lengths are taken from the BAM/CRAM or FASTA header without assuming a 24-chromosome human karyotype.
- **Canonical Filtering**: By default, canonical chromosomes (`1`–`22`, `X`, `Y`) are retained and mitochondrial, decoy, unplaced, and ALT contigs are skipped. Pass `--all-contigs` to include all header contigs.
- **Schema `2.0.0` Parquet Artifacts**: Sample and reference `.ccnv` files are Zstandard-compressed Apache Arrow Parquet tables validated against schema version `2.0.0` before computation.

## 1. `extract`

Streams alignments per contig, filters reads in deterministic precedence order, bins read starts with `numpy.bincount`, and writes `<prefix>.ccnv`. Effective read counters (`effective_reads_examined`, `effective_reads_kept`, `autosomal_reads_kept`, `sex_reads_kept`) record metrics strictly across binned target contigs.

```mermaid
flowchart TD
    InAlign["BAM or CRAM alignment"] --> CheckIdx{"Index (.bai/.crai) present?"}
    CheckIdx -- No --> BuildIdx["Generate index with pysam"]
    CheckIdx -- Yes --> OpenAlign["Open alignment file"]
    BuildIdx --> OpenAlign
    InAlign --> CheckCRAM{"Is CRAM?"}
    CheckCRAM -- Yes --> CheckFai["Require indexed FASTA (--fasta)"]
    CheckFai --> OpenAlign
    CheckCRAM -- No --> OpenAlign
    OpenAlign --> ParseHeader["Read header contigs & lengths"]
    ParseHeader --> LayoutFilter["Select canonical contigs (unless --all-contigs)"]
    LayoutFilter --> Budget["Split --threads into workers x decode_threads"]
    Budget --> Stream["Stream contigs in parallel"]
    Stream --> Filter["Filter reads & increment QC counters"]
    Filter --> Binning["Vectorized bin counting (np.bincount)"]
    Binning --> WriteParquet["Write <prefix>.ccnv (SAMPLE_SCHEMA 2.0.0)"]
```

- **Read Filter Precedence**: Reads are checked in order: `filter_unmapped`, `filter_no_start`, `filter_secondary`, `filter_supplementary`, `filter_duplicate`, `filter_improper_pair`, `filter_mapq`. The first failing check increments its QC counter and drops the read.
- **Thread Budget**: `--threads` splits CPU cores across contig worker processes and per-handle HTSlib decompression threads (`workers * decode_threads <= threads`).
- **Schema**: Columns `contig: string`, `start: int64`, `end: int64`, `count: float64` (or `uint64` in extracted samples), plus JSON metadata (`bin_size`, `contig_lengths`, `qc`, `filters`).

## 2. `reference`

Constructs an autosomal and optional sex-chromosome background model from a cohort samplesheet (`--samplesheet`) or a flat uniform-depth reference from a FASTA file (`--fasta`). Supports `--mode germline | prenatal` (default `germline`). In prenatal mode, also fits non-negative `ElasticNetCV` `FFY` (5 kb chrY), coarsens in memory to 100 kb for convex logit-link `RidgeCV`/`ElasticNetCV` ensemble `PREFACE` and robust Huber `FFX`, and persists `prenatal_autosomal` and `prenatal_ffy` branches alongside `ccnv.prenatal` metadata.

```mermaid
flowchart TD
    InputArg["Input: --samplesheet or --fasta"] --> ModeDecision{"Single FASTA provided?"}
    ModeDecision -- Yes --> FlatMode["Flat Mode: uniform depth + FASTA GC & N-masking"]
    ModeDecision -- No --> CohortMode["Cohort Mode: load & validate sample Parquets from samplesheet"]
    CohortMode --> RebinCheck["Rebin in memory via Polars integer fast-path or cumulative knots"]
    FlatMode --> CovTrack["Load optional --gc & --mappability tracks"]
    RebinCheck --> CovTrack
    CovTrack --> DepthNorm["DESeq2 median-of-ratios depth scaling, mask low-mappability/N bins & joint B-spline HuberRegressor covariate correction"]
    DepthNorm --> SexInference["Infer sex from signal contig coverage or prenatal 1D 2-GMM"]
    SexInference --> BranchPartition["Build autosomal, female_sex, and male_sex branches"]
    BranchPartition --> FitPCA["Fit Gavish-Donoho SVHT PCA & subtract systemic bias"]
    FitPCA --> FitNearest["Select --nearest-bins off-contig neighbors via signal-subspace KDTree"]
    FitNearest --> FitNull["Compute Smyth Empirical Bayes squeeze_var null_mean and null_std"]
    FitNull --> PrenatalCheck{"--mode prenatal?"}
    PrenatalCheck -- Yes --> FitPrenatal["Fit ElasticNetCV FFY, coarsened 100 kb logit RidgeCV/ElasticNetCV PREFACE & Huber FFX branches"]
    PrenatalCheck -- No --> WriteRef["Write <prefix>.ccnv (REFERENCE_SCHEMA 2.0.0)"]
    FitPrenatal --> WriteRef
    WriteRef --> WriteQC["Write <prefix>_qc.json (unless --no-qc-report)"]
```

- **In-Memory Rebinning**: Control samples extracted at a finer or coarser resolution are rebinned to `--bin-size` via mass-conserving geometric overlap when contig lengths match, utilizing a Rust-parallelized Polars integer-multiple fast path when coarsening.
- **Covariate Masking & Bias Removal**: Optional `--mappability` BigWig (where k-mer length should roughly match average sequencing read length) and `--gc` (FASTA, BigWig, or BED/TSV; derived automatically from the input FASTA in flat mode) tracks mask bins with mappability below `--min-mappability` (default `0.5`) or `>50%` `N` bases. Depth is scaled via DESeq2 median-of-ratios size factors (Anders & Huber, 2010), and a joint natural cubic B-spline basis with GC $\times$ mappability interaction is fitted per sample via `HuberRegressor` ($\epsilon = 1.35$) to divide out multiplicative GC wave and mappability bias before PCA.
- **Sex Inference & Branches**: Coverage on `--sex-signal-contig` (`chrY`/`Y`) separates female and male controls automatically (or via `--sex-cutoff` in germline mode, or via 1D 2-GMM and samplesheet labels in prenatal mode). The `autosomal` branch models autosomes across all controls; `female_sex` and `male_sex` model sex contigs for each group when at least `--min-sex-group-size` samples are present.
- **Between- & Within-Sample Modeling**: Singular Value Hard Thresholding (Gavish & Donoho, 2014) automatically determines the optimal PCA rank $P$, separating systemic technical batch effects from Marchenko-Pastur bulk noise. Each bin is embedded into a $(P+2)$-dimensional signal subspace and queried via C-compiled `scipy.spatial.KDTree` to select the `--nearest-bins` off-contig reference partners in $O(U \log U)$ time. Degrees-of-freedom-corrected residuals are moderated using Smyth (2004) Empirical Bayes variance shrinkage (`squeeze_var`) to parameterize per-bin `null_mean` and `null_std`.
- **Schema**: Columns `branch`, `contig`, `start`, `end`, `masked`, `gc`, `mappability`, `pca_mean`, `pca_loadings`, `nearest_ref_indices` (`DELTA_BINARY_PACKED`), `mean_distance`, `null_mean`, and `null_std`. Branches include `autosomal`, `female_sex`, `male_sex`, `prenatal_autosomal`, and `prenatal_ffy`.

## 3. `ref-qc`

Evaluates reference usability from nearest-neighbor distance distributions, masked bin fractions, and sex-signal contig coverage, writing `<prefix>_qc.json` and emitting a summary verdict (`PASS`, `WARN`, `FAIL`) to `stderr`.

- **Autosomal / Female Branches**: `FAIL` if no unmasked bins exist or `std_of_means > 10.0`; `WARN` if `std_of_means > 2.0`, outlier bins exceed `1.0%`, or bins have fewer than `min_ref_bins` neighbors.
- **Male Branch**: Also checks the sex-signal contig (`FAIL` if usable bins `< 20%` or mean distance `> 100.0`; `WARN` if usable bins `< 50%` or mean distance `> 5.0`).
- **Flat FASTA Reference**: Emits `WARN` noting that empirical cohort noise models and sex branches are absent.

## 4. `predict`

Projects a sample onto the reference model, normalizes coverage against within-sample off-contig reference bins, segments per contig using pre-split euchromatic change-point detection, and calls `GAIN`/`LOSS` events. Supports `--mode germline | prenatal` (default `germline`). In prenatal mode, normalizes `chrX` against the diploid `XX` (`female_sex`) reference branch, computes `FFY`, `FFX`, and `PREFACE`, classifies sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`), selects `active_ff`, annotates segments with expected ratios and discordance flags, and writes `<prefix>_qc.json`.

```mermaid
flowchart TD
    InSample["Sample Parquet (<sample>.ccnv)"] --> LoadCompat["Validate schema 2.0.0 & contig compatibility"]
    InRef["Reference Parquet (<reference>.ccnv)"] --> LoadCompat
    LoadCompat --> RebinCheck["Rebin sample counts in memory if bin sizes differ"]
    RebinCheck --> BranchSelect["Select autosomal and sex branch (XX for prenatal mode)"]
    BranchSelect --> ProjectPCA["Scale depth, remove GC/mappability bias, project onto PCA basis via HuberRegressor & remove systemic bias"]
    ProjectPCA --> MatchNearest["Compute log2 ratio against median of nearest off-contig bins via vectorized 2D neighbor matrix"]
    MatchNearest --> IterativeMask["Mask extreme outlier bins (--mask-repeats) & compute Empirical Bayes weights"]
    IterativeMask --> SegEngine["Pre-split euchromatic arm blocks & segment with ruptures (PELT / KernelCPD / Binseg)"]
    SegEngine --> CallEvents["Call GAIN / LOSS via Polars autocorrelation-corrected Stouffer Z-scores"]
    CallEvents --> PrenatalAnnot["2D Stoichiometric Bayesian SCA, precision-fused active_ff & LLR discordance (if --mode prenatal)"]
    PrenatalAnnot --> Outputs["Write bgzipped/tabix BED & VCFv4.2 + _qc.json + interactive HTML plot"]
```

- **Precision Weighting & Segmentation**: During single-sample prediction, counts are projected onto reference PCA loadings via robust Huber M-estimation (`HuberRegressor`, $\epsilon = 1.35$), preventing CNV-carrying chromosomes from distorting the diploid baseline. Within-sample references are refined through vectorized 2D neighbor-matrix median evaluations across $M$ iterations (default 3), masking outlier bins with $|Z_i| \ge 2.326347874$. Each usable bin receives an Empirical Bayes inverse-variance precision weight $w_i = \operatorname{clip}(s_0^2 / \tilde{\sigma}_{\text{null}, i}^2, 0.05, 20.0)$ proportional to the Gaussian log-likelihood (or $1/\bar{d}_i$ for flat FASTA references). Contigs are pre-split at masked gaps $>2\text{ Mb}$ into contiguous euchromatic arm blocks before change-point optimization using C-compiled `ruptures.KernelCPD` (or exact `Pelt` / `Binseg` with `WeightedL2Cost`). Segments are called using Polars-native Stouffer precision Z-scores with Newey-West AR(1) spatial autocorrelation correction, scaling sensitivity with $\sqrt{N}$ while preventing false-positive inflation from local genomic waves. In `--mode prenatal`, the 2D stoichiometric Gaussian Bayes classifier classifies sex chromosome aneuploidies with a Mahalanobis $\chi^2$ gate, independent estimators are combined via Bates-Granger precision fusion for `active_ff`, and segment discordance is scored via fetal-fraction-parameterized log-likelihood ratios ($\operatorname{LLR}_{\text{fetal}}$).
- **Outputs**: Writes natural-sorted, bgzip-compressed, tabix-indexed `.bed.gz` and `.vcf.gz` files (`_bins`, `_segments`, `_calls`), prediction quality control report `<prefix>_qc.json`, and an interactive Plotly genome profile `<prefix>_plot.html` (with optional cytoband ideogram track via `--cytobands`, prenatal status badge, and `#9333ea` discordant traces in prenatal mode).
