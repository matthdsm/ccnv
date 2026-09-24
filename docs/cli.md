# CLI reference

The `ccnv` entry point is a Typer app with three subcommands: `convert`,
`reference`, and `predict`. Commands take `--prefix` and append canonical
suffixes themselves; do not pass output file extensions.

Every command accepts `-h` or `--help`. File arguments and options marked as
existing paths are rejected by Typer before the stage runs.

`--loglevel` is accepted on every command (`DEBUG`, `INFO`, `WARNING`, `ERROR`,
`CRITICAL`; default `WARNING`). Pipeline steps emit `DEBUG` messages; raise the
level to see them on stderr.

For full algorithmic details, data transformations, and decision paths, see the [Pipeline Architecture & Subcommands](pipeline.md) guide.

## convert

Convert one BAM or CRAM into canonical sample Parquet. If the alignment has no
index, `convert` builds one with pysam (`.bai`/`.crai`).

```console
ccnv convert ALIGNMENT [--prefix PREFIX] [options]
```

| Flag / argument | Default | Meaning |
| --- | --- | --- |
| `ALIGNMENT` | required | BAM or CRAM. A missing index is created automatically. CRAM also needs `--fasta`. |
| `--prefix` | alignment stem | Output prefix. Writes `<prefix>.ccnv`. |
| `--bin-size` | `5000` | Target bin size in bases. The last bin on each contig may be shorter. |
| `--fasta` | unset | Reference FASTA required for CRAM. Must exist. |
| `--min-mapq` | `1` | Minimum mapping quality to keep. |
| `--keep-duplicates` | off | Keep SAM duplicate reads. |
| `--keep-secondary` | off | Keep secondary alignments. |
| `--keep-supplementary` | off | Keep supplementary alignments. |
| `--allow-improper-pairs` | off | Keep paired reads that are not proper pairs. Unpaired reads are not subject to the proper-pair rule. |
| `--all-contigs` | off | Bin every header contig instead of canonical chromosomes only. |
| `-t`, `--threads` | CPU count | CPU budget split between contig worker processes and per-handle BAM/CRAM decode threads. `1` bins serially with a single htslib thread. A budget of 8 on many contigs uses 4 workers × 2 decode threads. |
| `--loglevel` | `WARNING` | Logging verbosity for convert steps. |

Only canonical chromosomes are binned: autosomes 1-22 plus X and Y, written as UCSC (`chr1`, `chrX`) or plain NCBI/Ensembl (`1`, `X`) accessions. The `chr` prefix and the letter case are both ignored, so `CHR7` and `y` are canonical.

Everything else is skipped: mitochondrial contigs (`chrM`, `MT`),
`_alt`, `_decoy`, and `_random` scaffolds, unplaced `chrUn_*` scaffolds, and
`chrEBV`. Skipped contigs are absent from the bins, from `contig_lengths`, and
from `reference_identity.contig_order`, and their reads are never read. Convert
fails when a header has no canonical chromosome.

Pass `--all-contigs` to bin every contig in the header instead, including
mitochondrial and decoy sequence. This is the escape hatch for non-human or
custom layouts whose contigs do not use human chromosome names.
`ConvertConfig(all_contigs=...)` defaults to `False`, matching the CLI.

Contigs are binned in parallel across worker processes; output row order always
follows the alignment header. `--threads` is a CPU budget, not a raw process
count: convert keeps `workers × AlignmentFile(threads=)` at most that budget so
CRAM decompression is not stuck at pysam's default of one thread per handle.
`ConvertConfig(threads=...)` defaults to `1`, so library callers stay serial
unless they opt in.

```mermaid
flowchart LR
    A["BAM / CRAM Alignment"] --> B["pysam Indexing & Header Parsing"]
    B --> C["Canonical Contig Filtering"]
    C --> D["Multi-Process Contig Streaming"]
    D --> E["Flag & MAPQ Filtering"]
    E --> F["Vectorized Bin Counting"]
    F --> G["<prefix>.ccnv (Sample Parquet)"]
```

See [Pipeline Architecture: convert](pipeline.md#1-convert) for detailed processing logic and QC counters.

## reference

Build a reference model from control sample Parquet files, or a flat
uniform-depth model from a FASTA when no control samples are available.

```console
ccnv reference SAMPLES... --prefix PREFIX [options]
ccnv reference GENOME.fa --prefix PREFIX [--bin-size N]
```

| Flag / argument | Default | Meaning |
| --- | --- | --- |
| `SAMPLES...` | optional | Sample Parquet files and/or directories. Directories are searched recursively for sample Parquet; other files are ignored. Provide exactly one FASTA file when building a flat reference. |
| `--prefix` | required | Output prefix. Writes `<prefix>.ccnv`. |
| `--bin-size` | `5000` | Target bin size in bases. When building from samples, samples are rebinned to this target size (default 5000). When building from a FASTA, sets the uniform bin size. |
| `--all-contigs` | off | Include every FASTA contig in a flat reference instead of canonical chromosomes only. |
| `--autosomal-contig` | all non-sex contigs | Autosomal contig name. Repeat the flag for each contig. |
| `--sex-contig` | `X,Y,chrX,chrY` | Sex contig name. Repeat as needed. The default set is intersected with contigs present in the samples; any other explicit name must exist. |
| `--sex-signal-contig` | `chrY` | Contig whose coverage fraction is used to infer female vs male (higher fraction is treated as male). If `chrY` is absent, `Y` is used. |
| `--sex-cutoff` | unset | Coverage fraction of `--sex-signal-contig`. Samples above the cutoff are male. When omitted, a cutoff is inferred if two groups separate cleanly. |
| `--pca-components` | `5` | PCA components per branch. |
| `--nearest-bins` | `300` | Nearest off-contig reference bins per target bin. |
| `--max-batch-memory-mb` | `256` | Memory cap in MiB for batched distance computation. |
| `--min-cohort-size` | `10` | Minimum samples required for the autosomal branch. |
| `--min-sex-group-size` | `5` | Minimum inferred samples per sex before a sex-contig branch is written. |
| `--loglevel` | `WARNING` | Logging verbosity for reference steps. |

There is no `--sample-sex` flag. For human data, sex contigs default to
`X`/`Y`/`chrX`/`chrY` and the signal contig defaults to `chrY` or `Y`. Sex is
inferred from that signal, or from `--sex-cutoff` when you supply one. If
inference fails, only the autosomal branch is written.

Input samples are rebinned to the target bin size (default 5000) if their bin sizes
differ, provided their contig lengths match.

Providing a FASTA file writes a **flat** reference: bins follow the FASTA layout at
`--bin-size`, expected depth is proportional to bin width, and identical replica
profiles fill the usual PCA and nearest-bin model. Canonical chromosomes
including X and Y are treated as autosomal (no sex branches). This lets
`predict` run without a control cohort; it is not a substitute for matched
controls.

```mermaid
flowchart LR
    A["Control Parquets or FASTA"] --> B["Layout Validation & Rebinning"]
    B --> C["Depth Normalization"]
    C --> D["Sex Inference & Branch Partitioning"]
    D --> E["PCA Decomposition"]
    E --> F["Batched Nearest-Bin Search"]
    F --> G["<prefix>.ccnv (Reference Parquet)"]
```

See [Pipeline Architecture: reference](pipeline.md#2-reference) for cohort vs flat FASTA reference details.

## predict

Normalize one sample against a reference, segment, and write BED (and optional
plot).

```console
ccnv predict SAMPLE REFERENCE [--prefix PREFIX] [options]
```

| Flag / argument | Default | Meaning |
| --- | --- | --- |
| `SAMPLE` | required | Sample Parquet from `convert`. |
| `REFERENCE` | required | Reference Parquet from `reference`. |
| `--prefix` | sample stem | Output prefix for the files below. |
| `--sex` | `autosomal` | Sex branch: `autosomal`, `female`, or `male`. `autosomal` uses the same signal contig stored on the reference when possible. |
| `--penalty` | `1.0` | Weighted-SSE Binseg penalty. This is not DNAcopy `alpha`. |
| `--z-threshold` | `3.0` | Absolute segment z-score call cutoff. |
| `--ratio-threshold` | unset | Absolute log2-ratio call cutoff. When set, it overrides z-score calling. |
| `--min-reference-bins` | `3` | Minimum within-sample reference bins for a target bin. |
| `--mask-repeats` | `3` | Iterations of within-sample masking of extreme bins. |
| `--max-blacklist-gap` | `2000000` | Maximum masked gap (bases) that segmentation may bridge. |
| `--plot` / `--no-plot` | `--plot` | Write `<prefix>_plot.svg`. |
| `--loglevel` | `WARNING` | Logging verbosity for predict steps. |

Prediction writes:

- `<prefix>_bins.bed`
- `<prefix>_segments.bed`
- `<prefix>_calls.bed`
- `<prefix>_plot.svg` unless `--no-plot`
- `<prefix>_bins.ccnv` and `<prefix>_segments.ccnv`

BED coordinates are 0-based, half-open, with contig names from the sample
genome. Called segments also appear in `_calls.bed`.

If the sample has a different bin size than the reference model, it is
automatically rebinned to the reference bin coordinates in memory, provided their
contig lengths match. The reference Parquet file is never modified.

```mermaid
flowchart LR
    A["Sample & Reference Parquets"] --> B["In-Memory Rebinning (if needed)"]
    B --> C["Sex Branch Selection & Depth Scaling"]
    C --> D["PCA Projection & Bias Subtraction"]
    D --> E["Within-Sample Nearest-Bin Normalization"]
    E --> F["Iterative Outlier Masking"]
    F --> G["Weighted Ruptures Binseg"]
    G --> H["Z-score / Ratio Threshold Calling"]
    H --> I["BED, Parquet & SVG Outputs"]
```

See [Pipeline Architecture: predict](pipeline.md#3-predict) for detailed segmentation mechanics, calling criteria, and artifact specifications.