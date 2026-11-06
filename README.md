# Copynumber-CNV (`ccnv`) Blueprint

`ccnv` is an independent Python 3.12+ bioinformatics toolkit for copy-number variation (CNV) detection in shallow whole-genome sequencing (sWGS) data, inspired by the published statistical methodology of WISECONDOR (Straver et al., *Nucleic Acids Research*, 2014) and WisecondorX (Raman et al., *Nucleic Acids Research*, 2019).

- **Versioned Parquet Storage (`2.0.0`)**: Sample bin counts and reference models are stored as Zstandard-compressed Apache Arrow/Parquet `.ccnv` files with embedded QC, provenance, and optional prenatal model metadata.
- **Constitutional & Prenatal Analysis Modes**: Supports standard germline CNV profiling (`--mode germline`) as well as cell-free DNA (cfDNA) non-invasive prenatal screening (`--mode prenatal`) with pure NumPy/SciPy fetal fraction estimation (`FFY`, `FFX`, `PREFACE`), sex chromosome aneuploidy calling, and discordance filtering.
- **Organism-Agnostic & 0-Based Half-Open Coordinates**: Contig names, order, and lengths are read directly from alignment or FASTA headers without assuming a fixed 24-chromosome layout. All intervals are 0-based half-open `[start, end)`.
- **Pure-Python Execution**: Uses `pysam`, `pyBigWig`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, and weighted `ruptures` Binary Segmentation without heavy ML container dependencies.

## System Overview

The pipeline formalizes CNV identification across shallow whole-genome sequencing datasets into four discrete functional stages: read ingestion and binning (`convert`), cohort reference synthesis with covariate correction (`reference`), reference audit and quality verification (`ref-qc`), and single-sample projection and change-point segmentation (`predict`).

```mermaid
flowchart LR
    subgraph S1["1. Convert"]
        BAM["BAM / CRAM Alignment"] --> C["Sample Parquet (.ccnv)"]
    end
    subgraph S2["2. Reference"]
        SS["Samplesheet / FASTA"] --> REF["Reference Parquet (.ccnv)"]
        GC["GC / Mappability Tracks"] --> REF
        REF --> QC["Reference QC (.json)"]
    end
    subgraph S3["3. Ref-QC"]
        REF --> AUDIT["Quality Audit (PASS / WARN / FAIL)"]
    end
    subgraph S4["4. Predict"]
        C --> P["Segment & Call"]
        REF --> P
        P --> PQC["Predict QC (.json)"]
        P --> BED["BED Tables (.bed.gz)"]
        P --> VCF["VCFv4.2 (.vcf.gz)"]
        P --> HTML["Plotly Viewer (.html)"]
    end
```

### Mathematical & Algorithmic Architecture

#### 1. Alignment Ingestion & Proportional Rebinning
Alignments are streamed contig-by-contig with `pysam.AlignmentFile`, filtering reads failing QC (unmapped, secondary, supplementary, duplicate, vendor-failed, or mapping quality below threshold). Genomic intervals are partitioned into 0-based half-open bins $[s_i, e_i)$ of fixed width $B$ (default $5\,000\text{ bp}$), where terminal bins truncate at the contig boundary. When reconciling samples across heterogeneous bin sizes, counts are projected via proportional interval overlap matrix multiplication:
$$w_{ij} = \frac{\max(0, \min(e_i^{\text{src}}, e_j^{\text{tgt}}) - \max(s_i^{\text{src}}, s_j^{\text{tgt}}))}{e_i^{\text{src}} - s_i^{\text{src}}}$$
guaranteeing non-negative count conservation without bin interpolation artifacts.

#### 2. Robust Huber IRLS Covariate Bias Correction
Sequencing depth non-linearly covaries with regional base composition and mappability. Low-mappability bins ($< \tau_{\text{map}}$, default $0.5$) and bins with high ambiguous base fractions ($N > 0.5$) are masked. A standardized covariate basis matrix $\mathbf{X}$ is formulated (cubic polynomial for GC fraction, quadratic polynomial for mappability). Multiplicative biases are estimated using a 3-iteration Huber Iteratively Reweighted Least Squares (IRLS) regression:
$$\min_{\boldsymbol{\beta}} \sum_{i} \rho_{\delta}(y_i - \mathbf{x}_i^T \boldsymbol{\beta}), \quad \rho_{\delta}(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{if } |r| > \delta \end{cases}$$
with tuning threshold $\delta = 1.345 \cdot \text{MAD}(r)$. The estimated trend is clamped to $[0.20, 5.0]$ and divided out of per-sample bin depths, followed by per-sample total depth rescaling over usable bins.

#### 3. Off-Contig Nearest-Neighbor Reference Modeling
Following depth normalization and low-coverage masking ($\le 5\%$ of median bin depth), truncated Principal Component Analysis extracts $P$ deterministic components across cohort samples. For every target bin $i$, an empirical reference pool of $k$ bins (default 300) is selected. Crucially, bins on the same contig as $i$ are assigned infinite distance ($d_{ii} = \infty$) to prevent localized somatic copy-number alterations from dampening detection sensitivity:
$$d_{ij} = \sqrt{\sum_{p=1}^P (\mathbf{v}_{ip} - \mathbf{v}_{jp})^2}$$
In control cohorts, an empirical null model is parameterized over up to 100 samples by evaluating per-bin log2 ratio deviations against neighbor medians:
$$R_{s, i} = \log_2 \left( \frac{y_{s, i}}{\text{median}_{j \in \mathcal{N}_i}(y_{s, j})} \right)$$
storing empirical expectations $\mu_i$ and sample standard deviations $\sigma_i$ in the reference model.

#### 4. Iterative Within-Sample Filtering & Weighted Ruptures Segmentation
During prediction, the test sample is projected into PCA space to compute expected baseline depths. Within-sample reference bins are refined through $M$ iterations (default 2) by temporarily excluding bins exhibiting $|Z_i| \ge 2.3263$ (99th percentile of a standard normal distribution). Genome segmentation is framed as an optimization problem minimizing a penalized weighted residual sum of squares:
$$\mathcal{L}(\mathcal{T}) = \sum_{m=1}^{M+1} \mathrm{cost}(t_{m-1}, t_m) + \beta M, \quad \mathrm{cost}(s, e) = \sum_{i=s}^{e-1} w_i (y_i - \bar{y}_{w})^2$$
where weights $w_i = \sigma_i^{-2}$, $y_i = \log_2(\text{depth}_i / \text{expected}_i)$, and $\bar{y}_w = \frac{\sum w_i y_i}{\sum w_i}$. Contiguous unmapped tracts exceeding $\Delta_{\text{max}}$ (default $2\,000\,000\text{ bp}$) enforce segment breaks across centromeres and assembly gaps. Segments are called as gains or losses when $|\bar{Z}| \ge Z_{\text{thresh}}$ (default 3.0) and $|\bar{y}_w| \ge r_{\text{thresh}}$ (default 0.2).

#### 5. Prenatal Fetal Fraction Estimation & Sex Chromosome Analysis
In `--mode prenatal`, `ccnv` integrates non-invasive prenatal analysis directly into reference building and prediction without heavy machine learning dependencies. During `reference`, a 5 kb `chrY` non-negative Lasso regression (`FFY`) is calibrated alongside a 20x coarsened 100 kb autosomal PCA, full-batch Resilient Backpropagation Sigmoid neural network / Ordinary Linear Model (`PREFACE`), and robust Huber M-estimation on `chrX` (`FFX`). During `predict`, `chrX` is normalized against the diploid `XX` (`female_sex`) reference branch, fetal sex and sex chromosome aneuploidies (`XX`, `XY`, `XO`, `XXX`, `XXY`, `XYY`, `DISCORDANT_MIXED`) are classified, and active fetal fraction is selected. Predicted CNV segments are evaluated against expected prenatal copy ratios $\log_2(1 \pm 0.5 \cdot \text{active\_ff})$, flagging sub-fetal noise, super-fetal maternal changes, or gonosome shifts with dedicated discordance highlighting and structured `<prefix>_qc.json` reporting.

## Subdirectories & Key Modules

- `src/ccnv/cli.py`: The `ccnv` Typer entry point and subcommands (`convert`, `reference`, `ref-qc`, `predict`).
- `src/ccnv/convert.py`: Root orchestrator for BAM/CRAM read ingestion, canonical contig binning, and effective target-contig read QC.
- `src/ccnv/reference.py`: Root orchestrator for reference Parquet synthesis and cohort rebinning across germline and prenatal modes.
- `src/ccnv/ref_qc.py`: Root orchestrator for reference model quality auditing and usability verdicts.
- `src/ccnv/predict.py`: Root orchestrator for sample projection, within-sample normalization, Ruptures segmentation, and event calling across germline and prenatal modes.
- `src/ccnv/models/`: Layer 0 domain objects, configurations, results, versioned Arrow schemas (`2.0.0`), and exceptions.
- `src/ccnv/stats/`: Layer 1 pure mathematical, nearest-neighbor, PCA, regression, and prenatal kernels.
- `src/ccnv/io/`: Layer 2 file format readers (BAM/CRAM, BigWig, FASTA, cytoband BED/TSV), samplesheet parsing, and Parquet storage.
- `src/ccnv/features/`: Layer 3 feature transformations, dynamic rebinning, covariate bias correction, and prenatal inference models.
- `src/ccnv/qc/`: Layer 4 quality control auditors and structured JSON report builders.
- `src/ccnv/report/`: Layer 5 deliverable writers for bgzipped/tabix BED, VCFv4.2, and interactive Plotly HTML viewer (`templates/genome_plot.html.j2`).

## Global Invariants

- Schema Version: All serialized `.ccnv` Parquet files strictly conform to Arrow schema version `2.0.0` and validate against `ccnv.models.schemas:validate_table`.
- Coordinate System: All intervals across memory representations, Parquet tables, and output BED/VCF files are 0-based half-open `[start, end)`.
- Contig Authority: Contig names, order, and lengths from alignment headers or reference FASTA files are authoritative; arbitrary 24-chromosome layouts are forbidden.
- Off-Contig Neighbor Separation: Reference bin neighbors must never be drawn from the same contig as the target bin ($d_{\text{intra-contig}} = \infty$).
- Masked Bin Propagation: Bins flagged as masked during covariate or coverage filtering are propagated consistently through PCA projection and excluded from segmentation cost evaluation.
- Diploid XX Gonosome Baseline: In prenatal mode, `chrX` normalization is always evaluated against the diploid `XX` (`female_sex`) reference branch to accurately capture fetal sex-chromosome dosage changes.

## Global Side Effects

- Filesystem Persistence: Writes zstd-compressed Parquet tables (`.ccnv`), bgzipped and tabix-indexed BED tables (`.bed.gz`, `.tbi`), bgzipped VCF files (`.vcf.gz`, `.tbi`), quality metrics JSON (`<prefix>_qc.json` for both `reference` and `predict`), and standalone HTML visualization files (`.html`).
- Alignment Indexing: Automatically generates missing BAM (`.bai`), CRAM (`.crai`), or FASTA (`.fai`) index files in place on the filesystem.
- Process Concurrency: Spawns worker process pools (`ProcessPoolExecutor`) for parallel contig read binning, cohort rebinning, and batched nearest-neighbor Euclidean distance matrix computations.

## Installation

```console
python -m pip install .
ccnv --help
```

For development with [Pixi](https://pixi.sh):

```console
pixi install
pixi run ccnv --help
```

Multi-platform container images (`linux/amd64` and `linux/arm64`) are published to `ghcr.io/matthdsm/copynumber-cnv`:

```console
docker pull ghcr.io/matthdsm/copynumber-cnv:latest
docker run --rm ghcr.io/matthdsm/copynumber-cnv:latest --help
```

## Quickstart

```console
# 1. Convert BAM or CRAM alignments into binned sample Parquet (.ccnv)
ccnv convert sample.bam --prefix sample --bin-size 5000

# 2a. Build a germline reference model from a cohort samplesheet (with optional GC & mappability tracks)
ccnv reference -s cohort.yaml --prefix reference --bin-size 5000 --gc genome.fa --mappability mappability.bw

# 2b. Or build a flat uniform-depth germline reference from a FASTA when no controls are available
ccnv reference --fasta genome.fa --prefix reference --bin-size 5000 --mappability mappability.bw

# 2c. Or build a unified prenatal reference from a maternal cfDNA cohort samplesheet
ccnv reference -s cohort.yaml --mode prenatal --prefix prenatal_ref --bin-size 5000

# 3. Inspect reference quality control metrics
ccnv ref-qc reference.ccnv

# 4a. Predict copy-number gains and losses in constitutional/germline mode
ccnv predict sample.ccnv reference.ccnv --prefix results/sample

# 4b. Or run prenatal prediction with fetal fraction estimation and sex chromosome analysis
ccnv predict sample.ccnv prenatal_ref.ccnv --mode prenatal --prefix results/sample_prenatal
```

- `convert` writes `<prefix>.ccnv` (automatically indexing BAM/CRAM files when needed; CRAM inputs also require `--fasta`). Metadata records effective read counts and filter drops on target contigs.
- `reference` writes `<prefix>.ccnv` and `<prefix>_qc.json` from a cohort samplesheet (`-s/--samplesheet`) or flat FASTA (`--fasta`). In `--mode prenatal`, it fits multi-resolution `FFY`, `PREFACE`, and `FFX` models alongside 5 kb CNV branches.
- `ref-qc` audits `<prefix>.ccnv` and outputs `<prefix>_qc.json` with a usability verdict (`PASS`, `WARN`, `FAIL`).
- `predict` writes bgzipped and tabix-indexed BED and VCFv4.2 tables (`_bins`, `_segments`, `_calls`), an interactive genome viewer (`<prefix>_plot.html`), and a run quality metrics report (`<prefix>_qc.json`). In `--mode prenatal`, segments include prenatal scaling and discordance tags, the viewer highlights discordant calls in high contrast, and `_qc.json` details fetal fraction and aneuploidy calls.

## Documentation

- [CLI Reference](docs/cli.md): Complete command-line flags, defaults, and I/O specifications.
- [Pipeline Architecture](docs/pipeline.md): Data flow, coordinate contracts, and algorithmic stages.
- [Benchmarks](docs/benchmark.md): Concordance, storage efficiency, and segmentation benchmarks.

## System Verification

- Run: `pytest`

## Development

```console
pixi run format
pixi run lint
pixi run test
pixi run hooks
```

All changes must be submitted via pull request and recorded in [CHANGELOG.md](CHANGELOG.md).

## License

Licensed under the MIT License. See [LICENSE.md](LICENSE.md).
