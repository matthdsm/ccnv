# Copynumber-CNV (`ccnv`)

Copynumber-CNV (`ccnv`) is an independent Python 3.14+ bioinformatics toolkit for copy-number variation (CNV) detection in shallow whole-genome sequencing (sWGS) data, building on published within-sample reference normalization methodology (Straver et al., *Nucleic Acids Research*, 2014; Raman et al., *Nucleic Acids Research*, 2019).

!!! note

    Copynumber-CNV (`ccnv`) is in active pre-1.0 alpha development (`0.1.0dev`). CLI interfaces, data models, and schemas may evolve before the initial stable release.

---

## Key Features

- **Versioned Parquet Storage (`1.0.0`)**: Sample bin counts, per-bin cfDNA fragmentomic channels (`fragment_lengths`, `endpoint_count`, `frag_gc_sum`), and reference models are stored as Zstandard-compressed Apache Arrow/Parquet `.ccnv` files with embedded QC, `ccnv.fragmentomics`, and optional `ccnv.prenatal` metadata.
- **Constitutional & Prenatal Analysis Modes**: Supports germline CNV profiling (`--mode germline`) and cell-free DNA (cfDNA) non-invasive prenatal screening (`--mode prenatal`) with pure NumPy/SciPy/scikit-learn fetal fraction estimation (`FFY`, `FFX`, `FFA`, `FF_size`), Fisher-optimal fragment length read weighting, 2D stoichiometric Bayesian sex chromosome aneuploidy calling, and segment discordance and fragmentomic origin annotation.
- **Organism-Agnostic & 0-Based Half-Open Coordinates**: Contig names, order, and lengths are read directly from alignment or FASTA headers without assuming a fixed 24-chromosome layout. All intervals are 0-based half-open `[start, end)`.
- **Pure-Python & DataFrame-Native Execution**: Uses `polars`, `pysam`, `pyBigWig`, `numpy`, `scipy`, `pyarrow`, `scikit-learn`, and weighted `ruptures` segmentation without external R or container runtime dependencies.

---

## Pipeline Overview

The pipeline structures CNV identification across shallow whole-genome sequencing datasets into four stages: alignment ingestion and binning (`extract`), cohort or FASTA reference construction with covariate bias removal (`reference`), reference usability verification (`ref-qc`), and single-sample projection, change-point segmentation, and event calling (`predict`).

```mermaid
flowchart LR
    subgraph S1["1. Extract"]
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
        BAM -.-> P
        REF --> P
        P --> PQC["Predict QC (.json)"]
        P --> BED["BED Tables (.bed.gz)"]
        P --> VCF["VCFv4.2 (.vcf.gz)"]
        P --> HTML["Plotly Viewer (.html)"]
    end
```

---

## Documentation Sections

- [Pipeline Architecture](pipeline.md): Data flow, coordinate contracts, and algorithmic stages.
- [CLI Reference](cli.md): Command-line options, defaults, input requirements, deliverables, and exit codes.
- [Mathematical Methodology](methodology.md): Mathematical formulations, statistical assumptions, model selections, and constant provenance.

---

## Quickstart

```console
# 1. Extract BAM or CRAM alignments into binned sample Parquet (.ccnv)
ccnv extract sample.bam --prefix sample --bin-size 5000

# 2a. Build a germline reference model from a cohort samplesheet
ccnv reference -s cohort.yaml --prefix reference --bin-size 5000 --gc genome.fa --mappability mappability.bw

# 2b. Or build a flat uniform-depth germline reference from a FASTA
ccnv reference --fasta genome.fa --prefix reference --bin-size 5000 --mappability mappability.bw

# 2c. Or build a unified prenatal reference from a maternal cfDNA cohort samplesheet
ccnv reference -s cohort.yaml --mode prenatal --prefix prenatal_ref --bin-size 5000

# 3. Audit reference quality control metrics
ccnv ref-qc reference.ccnv

# 4a. Predict copy-number gains and losses from .ccnv, .bam, or .cram (with --fasta)
ccnv predict sample.ccnv reference.ccnv --prefix results/sample

# 4b. Or run prenatal prediction with fetal fraction estimation and sex chromosome analysis
ccnv predict sample.ccnv prenatal_ref.ccnv --mode prenatal --prefix results/sample_prenatal
```
