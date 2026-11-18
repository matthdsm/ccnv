"""Thin command-line interface for ccnv pipeline stages."""

import logging
import math
import os
import sys
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated, NoReturn

import typer

from ccnv.extract import run as run_extract
from ccnv.models import (
    DEFAULT_BIN_SIZE,
    DEFAULT_MIN_MAPPABILITY,
    AnalysisMode,
    CcnvError,
    ExtractConfig,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    ReferenceQCConfig,
    SexBranch,
)
from ccnv.predict import run as run_predict
from ccnv.ref_qc import run_reference_qc
from ccnv.reference import run as run_reference

app = typer.Typer(
    name="ccnv",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

DEFAULT_THREADS = os.cpu_count() or 1
cli_logger = logging.getLogger("ccnv.cli")


class LogLevel(StrEnum):
    """Python logging levels accepted by ``--loglevel``."""

    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


def configure_logging(level: LogLevel) -> None:
    """Send ``ccnv`` logs to stderr at the requested level."""

    logger = logging.getLogger("ccnv")
    logger.setLevel(level.value)
    logger.propagate = False
    handler = next(
        (item for item in logger.handlers if item.name == "ccnv"),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.name = "ccnv"
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    elif isinstance(handler, logging.StreamHandler):
        handler.stream = sys.stderr
    handler.setLevel(level.value)


LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        "--loglevel",
        case_sensitive=False,
        help="Logging verbosity for pipeline steps",
    ),
]

ModeOption = Annotated[
    AnalysisMode,
    typer.Option(
        "--mode",
        case_sensitive=False,
        help="Analysis mode: germline constitutional CNV or prenatal NIPS cfDNA",
    ),
]

MappabilityOption = Annotated[
    Path | None,
    typer.Option(
        "--mappability",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "Mappability BigWig (.bw/.bigWig) track for covariate normalization "
            "(k-mer length should roughly match average sequencing read length)"
        ),
    ),
]

GcOption = Annotated[
    Path | None,
    typer.Option(
        "--gc",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="GC content track (.bw/.bigWig, .bed/.bed.gz, or .fa/.fasta/.fna)",
    ),
]


def _require_positive_float(value: float | None) -> float | None:
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise typer.BadParameter("must be a finite positive number (> 0)")
    return value


def _require_finite_unit_float(value: float | None) -> float | None:
    if value is not None and (not math.isfinite(value) or value < 0.0 or value > 1.0):
        raise typer.BadParameter("must be a finite number in [0, 1]")
    return value


MinMappabilityOption = Annotated[
    float,
    typer.Option(
        "--min-mappability",
        min=0.0,
        max=1.0,
        callback=_require_finite_unit_float,
        help="Minimum bin mappability in [0, 1] required to stay unmasked",
    ),
]

FastaOption = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Reference FASTA required for CRAM inputs",
    ),
]


@app.command()
def extract(
    alignment: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "BAM or CRAM input; single-threaded extraction streams without an "
                "index, while --threads > 1 requires a .bai, .csi, or .crai index"
            ),
        ),
    ],
    prefix: Annotated[
        Path | None,
        typer.Option(help="Output path prefix; writes PREFIX.ccnv"),
    ] = None,
    bin_size: Annotated[
        int, typer.Option(min=1, help="Target bin size in bases")
    ] = DEFAULT_BIN_SIZE,
    fasta: FastaOption = None,
    min_mapq: Annotated[
        int,
        typer.Option("--min-mapq", min=0, help="Minimum mapping quality to keep"),
    ] = 1,
    keep_duplicates: Annotated[
        bool, typer.Option("--keep-duplicates", help="Keep SAM duplicate reads")
    ] = False,
    keep_secondary: Annotated[
        bool, typer.Option("--keep-secondary", help="Keep secondary alignments")
    ] = False,
    keep_supplementary: Annotated[
        bool,
        typer.Option("--keep-supplementary", help="Keep supplementary alignments"),
    ] = False,
    allow_improper_pairs: Annotated[
        bool,
        typer.Option(
            "--allow-improper-pairs",
            help="Keep paired reads that are not proper pairs",
        ),
    ] = False,
    all_contigs: Annotated[
        bool,
        typer.Option(
            "--all-contigs",
            help="Bin every header contig instead of canonical chromosomes only",
        ),
    ] = False,
    threads: Annotated[
        int,
        typer.Option(
            "-t",
            "--threads",
            min=1,
            help=(
                "CPU budget for parallel 50 Mb tile streaming and BAM/CRAM "
                "decompression (requires alignment index when > 1)"
            ),
        ),
    ] = DEFAULT_THREADS,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Extract binned counts from one alignment into canonical sample Parquet."""

    configure_logging(loglevel)
    cli_logger.debug(
        "starting extract: alignment=%s, bin_size=%d, threads=%d",
        alignment,
        bin_size,
        threads,
    )
    started = perf_counter()
    try:
        config = ExtractConfig(
            alignment,
            prefix if prefix is not None else alignment.with_suffix(""),
            bin_size,
            fasta=fasta,
            filters=ReadFilters(
                min_mapping_quality=min_mapq,
                exclude_duplicates=not keep_duplicates,
                exclude_secondary=not keep_secondary,
                exclude_supplementary=not keep_supplementary,
                require_proper_pair=not allow_improper_pairs,
            ),
            threads=threads,
            all_contigs=all_contigs,
        )
        run_extract(config)
        cli_logger.debug("extract completed in %.3f s", perf_counter() - started)
    except (CcnvError, ValueError, OSError) as error:
        _fail(error, 1)


@app.command()
def reference(
    prefix: Annotated[
        Path,
        typer.Option(help="Output path prefix; writes PREFIX.ccnv"),
    ],
    samplesheet: Annotated[
        Path | None,
        typer.Option(
            "-s",
            "--samplesheet",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Root-list samplesheet (.yaml, .yml, or .json) describing the cohort",
        ),
    ] = None,
    fasta: Annotated[
        Path | None,
        typer.Option(
            "--fasta",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Reference FASTA for flat reference building",
        ),
    ] = None,
    mode: ModeOption = AnalysisMode.germline,
    prenatal_binsize: Annotated[
        int,
        typer.Option(
            "--prenatal-binsize",
            min=1,
            help="Bin size in bases for prenatal models (default: 100000)",
        ),
    ] = 100_000,
    bin_size: Annotated[
        int, typer.Option(min=1, help="Target bin size in bases")
    ] = DEFAULT_BIN_SIZE,
    all_contigs: Annotated[
        bool,
        typer.Option(
            "--all-contigs",
            help="Include every FASTA contig in a flat reference",
        ),
    ] = False,
    autosomal_contig: Annotated[
        list[str] | None,
        typer.Option(
            "--autosomal-contig",
            help="Autosomal contig name; repeat as needed",
        ),
    ] = None,
    sex_contig: Annotated[
        list[str] | None,
        typer.Option("--sex-contig", help="Sex contig name; repeat as needed"),
    ] = None,
    sex_signal_contig: Annotated[
        str | None,
        typer.Option(help="Contig used for inferred sex; higher fraction is male"),
    ] = None,
    sex_cutoff: Annotated[
        float | None,
        typer.Option(
            "--sex-cutoff",
            min=0.0,
            max=1.0,
            callback=_require_finite_unit_float,
            help="Signal-contig coverage fraction cutoff; samples above it are male",
        ),
    ] = None,
    pca_components: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Maximum PCA components per branch (defaults to dynamic SVHT rank)",
        ),
    ] = None,
    max_batch_memory_mb: Annotated[
        int,
        typer.Option(min=1, help="Maximum memory in MiB for distance batches"),
    ] = 256,
    min_cohort_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum samples required for the autosomal branch",
        ),
    ] = 10,
    min_sex_group_size: Annotated[
        int, typer.Option(min=1, help="Minimum samples per sex for a sex-contig branch")
    ] = 5,
    qc_report: Annotated[
        bool,
        typer.Option(
            "--qc-report/--no-qc-report",
            help="Generate reference quality control report",
        ),
    ] = True,
    mappability: MappabilityOption = None,
    gc: GcOption = None,
    min_mappability: MinMappabilityOption = DEFAULT_MIN_MAPPABILITY,
    threads: Annotated[
        int,
        typer.Option(
            "-t",
            "--threads",
            min=1,
            help="CPU budget for parallel loading and reference modeling",
        ),
    ] = DEFAULT_THREADS,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Build a reference model from control samples or a flat FASTA layout."""

    configure_logging(loglevel)
    cli_logger.debug(
        "starting reference: prefix=%s, bin_size=%d, mode=%s, threads=%d",
        prefix,
        bin_size,
        mode.value,
        threads,
    )
    started = perf_counter()
    if samplesheet is None and fasta is None:
        _fail(ValueError("provide exactly one of --samplesheet or --fasta"), 2)
    if samplesheet is not None and fasta is not None:
        _fail(ValueError("provide either --samplesheet or --fasta, not both"), 2)
    try:
        run_reference(
            ReferenceConfig(
                all_contigs=all_contigs,
                autosomal_contigs=(
                    tuple(autosomal_contig) if autosomal_contig else None
                ),
                bin_size=bin_size,
                fasta=fasta,
                gc=gc,
                mappability=mappability,
                max_batch_memory_mb=max_batch_memory_mb,
                min_cohort_size=min_cohort_size,
                min_mappability=min_mappability,
                min_sex_group_size=min_sex_group_size,
                mode=mode,
                pca_components=pca_components,
                prefix=prefix,
                prenatal_bin_size=prenatal_binsize,
                qc_report=qc_report,
                samplesheet_path=samplesheet,
                sex_contigs=(tuple(sex_contig) if sex_contig is not None else None),
                sex_cutoff=sex_cutoff,
                sex_signal_contig=sex_signal_contig,
                threads=threads,
            )
        )
        cli_logger.debug("reference completed in %.3f s", perf_counter() - started)
    except (CcnvError, ValueError, OSError) as error:
        _fail(error, 1)


@app.command(name="ref-qc")
def ref_qc(
    reference: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Reference Parquet file (.ccnv)",
        ),
    ],
    prefix: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Prefix for report output; writes PREFIX_qc.json (defaults to "
                "reference path without suffix)"
            ),
        ),
    ] = None,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Analyze an existing reference model and evaluate quality control metrics."""

    configure_logging(loglevel)
    cli_logger.debug("starting ref-qc: reference=%s", reference)
    started = perf_counter()
    try:
        run_reference_qc(ReferenceQCConfig(reference=reference, prefix=prefix))
        cli_logger.debug("ref-qc completed in %.3f s", perf_counter() - started)
    except (CcnvError, ValueError, OSError) as error:
        _fail(error, 1)


@app.command()
def predict(
    sample: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="BAM, CRAM, or sample Parquet input",
        ),
    ],
    reference: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Reference Parquet input",
        ),
    ],
    prefix: Annotated[
        Path | None,
        typer.Option(
            "--prefix",
            "-o",
            help="Output path prefix for BED, VCF, and HTML deliverables",
        ),
    ] = None,
    sex: Annotated[
        SexBranch,
        typer.Option(help="Sex branch used for gonosome models"),
    ] = SexBranch.autosomal,
    penalty: Annotated[
        float,
        typer.Option(
            callback=_require_positive_float,
            help="Breakpoint penalty multiplier (gamma) for calibrated mBIC segmentation",
        ),
    ] = 1.0,
    z_threshold: Annotated[
        float,
        typer.Option(
            "--z-threshold",
            callback=_require_positive_float,
            help="Absolute segment z call cutoff",
        ),
    ] = 3.0,
    ratio_threshold: Annotated[
        float | None,
        typer.Option(
            "--ratio-threshold",
            callback=_require_positive_float,
            help="Absolute log2-ratio call cutoff; overrides z-score calling",
        ),
    ] = None,
    min_reference_bins: Annotated[
        int,
        typer.Option(
            "--min-reference-bins",
            min=1,
            help="Minimum within-sample reference bins",
        ),
    ] = 3,
    mask_repeats: Annotated[
        int,
        typer.Option(
            "--mask-repeats",
            min=1,
            help="Within-sample masking iterations",
        ),
    ] = 3,
    max_blacklist_gap: Annotated[
        int,
        typer.Option(
            "--max-blacklist-gap",
            min=0,
            help="Maximum masked gap bridged during segmentation",
        ),
    ] = 2_000_000,
    plot: Annotated[
        bool,
        typer.Option("--plot/--no-plot", help="Write an interactive genome HTML plot"),
    ] = True,
    cytobands: Annotated[
        Path | None,
        typer.Option(
            "--cytobands",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=(
                "Cytoband BED/TSV file (.txt/.bed/.gz) to visualize above the plot "
                "(overrides bundled GRCh38 cytobands)"
            ),
        ),
    ] = None,
    vcf: Annotated[
        bool,
        typer.Option(
            "--vcf/--no-vcf",
            help="Write VCFv4.2 outputs for segments and calls (use --bins-vcf for per-bin VCF)",
        ),
    ] = True,
    bins_vcf: Annotated[
        bool,
        typer.Option(
            "--bins-vcf/--no-bins-vcf",
            help="Write bin-level VCFv4.2 output when --vcf is enabled",
        ),
    ] = False,
    threads: Annotated[
        int,
        typer.Option(
            "--threads",
            "-t",
            min=1,
            help="Worker threads for direct BAM/CRAM streaming and deliverable generation",
        ),
    ] = DEFAULT_THREADS,
    fasta: FastaOption = None,
    mappability: MappabilityOption = None,
    gc: GcOption = None,
    min_mappability: MinMappabilityOption = DEFAULT_MIN_MAPPABILITY,
    mode: ModeOption = AnalysisMode.germline,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Predict copy-number events for one sample."""

    configure_logging(loglevel)
    cli_logger.debug(
        "starting predict: sample=%s, reference=%s, sex=%s, mode=%s, threads=%d",
        sample,
        reference,
        sex.value,
        mode.value,
        threads,
    )
    started = perf_counter()
    try:
        run_predict(
            PredictConfig(
                sample=sample,
                reference=reference,
                prefix=prefix if prefix is not None else sample.with_suffix(""),
                sex=sex,
                penalty=penalty,
                z_threshold=z_threshold,
                ratio_threshold=ratio_threshold,
                min_reference_bins=min_reference_bins,
                mask_repeats=mask_repeats,
                max_blacklist_gap=max_blacklist_gap,
                plot=plot,
                vcf=vcf,
                bins_vcf=bins_vcf,
                cytobands=cytobands,
                mappability=mappability,
                gc=gc,
                fasta=fasta,
                min_mappability=min_mappability,
                mode=mode,
                threads=threads,
            )
        )

        cli_logger.debug("predict completed in %.3f s", perf_counter() - started)
    except (CcnvError, ValueError, OSError) as error:
        _fail(error, 1)


def _fail(error: BaseException, code: int) -> NoReturn:
    """Print error message to stderr and terminate CLI with given exit code."""
    typer.secho(str(error), err=True, fg="red")
    raise typer.Exit(code=code) from error


if __name__ == "__main__":
    app()
