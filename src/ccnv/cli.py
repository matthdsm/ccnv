"""Thin command-line interface for ccnv pipeline stages."""

import logging
import os
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated, NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from ccnv.convert import ConvertError
from ccnv.convert import run as run_convert
from ccnv.models import (
    DEFAULT_BIN_SIZE,
    ConvertConfig,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    ReferenceQCConfig,
)
from ccnv.predict import PredictError
from ccnv.predict import run as run_predict
from ccnv.reference import (
    DEFAULT_SEX_CONTIGS,
    DEFAULT_SEX_SIGNAL_CONTIGS,
    ReferenceError,
)
from ccnv.reference import (
    run as run_reference,
)
from ccnv.reference_qc import (
    ReferenceQCError,
    run_reference_qc,
)
from ccnv.schemas import ArtifactKind, SchemaError, decode_metadata

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


class SexBranch(StrEnum):
    """Allowed predict sex-branch values."""

    autosomal = "autosomal"
    female = "female"
    male = "male"


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


class SampleInputError(ValueError):
    """Raised when reference sample arguments cannot be expanded."""


def expand_sample_inputs(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Expand files and directories into unique sample Parquet paths."""

    samples: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            matches = [
                candidate.resolve()
                for candidate in sorted(resolved.rglob("*.ccnv"))
                if _is_sample_parquet(candidate)
            ]
            if not matches:
                raise SampleInputError(f"{path} contains no sample CCNV files")
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    samples.append(match)
            continue
        if resolved not in seen:
            seen.add(resolved)
            samples.append(resolved)
    return tuple(samples)


def _is_sample_parquet(path: Path) -> bool:
    try:
        metadata = decode_metadata(pq.read_schema(path).metadata)
    except (OSError, TypeError, ValueError, SchemaError, pa.ArrowException):
        return False
    return metadata.get("artifact_kind") == ArtifactKind.SAMPLE.value


@app.command()
def convert(
    alignment: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="BAM or CRAM input; a missing index is created automatically",
        ),
    ],
    prefix: Annotated[
        Path | None,
        typer.Option(help="Output path prefix; writes PREFIX.ccnv"),
    ] = None,
    bin_size: Annotated[
        int, typer.Option(min=1, help="Target bin size in bases")
    ] = DEFAULT_BIN_SIZE,
    fasta: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Reference FASTA required for CRAM inputs",
        ),
    ] = None,
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
                "CPU budget split between contig worker processes and "
                "per-handle BAM/CRAM decode threads"
            ),
        ),
    ] = DEFAULT_THREADS,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Convert one alignment into canonical binned counts."""

    configure_logging(loglevel)
    cli_logger.debug(
        "starting convert: alignment=%s, bin_size=%d, threads=%d",
        alignment,
        bin_size,
        threads,
    )
    started = perf_counter()
    config = ConvertConfig(
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
    try:
        run_convert(config)
        cli_logger.debug("convert completed in %.3f s", perf_counter() - started)
    except ConvertError as error:
        _fail(error, 1)


@app.command()
def reference(
    prefix: Annotated[
        Path,
        typer.Option(help="Output path prefix; writes PREFIX.ccnv"),
    ],
    samples: Annotated[
        list[Path] | None,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help=(
                "Sample Parquet files and/or directories; directories are searched"
                " recursively for sample Parquet."
                "To build a flat reference, provide exactly one fasta file."
            ),
        ),
    ] = None,
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
        list[str],
        typer.Option("--sex-contig", help="Sex contig name; repeat as needed"),
    ] = DEFAULT_SEX_CONTIGS,
    sex_signal_contig: Annotated[
        str,
        typer.Option(help="Contig used for inferred sex; higher fraction is male"),
    ] = DEFAULT_SEX_SIGNAL_CONTIGS[0],
    sex_cutoff: Annotated[
        float | None,
        typer.Option(
            "--sex-cutoff",
            min=0.0,
            max=1.0,
            help="Signal-contig coverage fraction cutoff; samples above it are male",
        ),
    ] = None,
    pca_components: Annotated[
        int, typer.Option(min=1, help="PCA components per branch")
    ] = 5,
    nearest_bins: Annotated[
        int,
        typer.Option(
            min=1,
            help="Nearest off-contig reference bins per target",
        ),
    ] = 300,
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
        "starting reference: prefix=%s, bin_size=%d, threads=%d",
        prefix,
        bin_size,
        threads,
    )
    started = perf_counter()
    sample_paths: tuple[Path, ...] = ()
    fasta: Path | None = None
    if samples:
        if len(samples) == 1 and samples[0].suffix.lower() in (".fasta", ".fa", ".fna"):
            fasta = samples[0]
        else:
            try:
                sample_paths = expand_sample_inputs(samples)
            except SampleInputError as error:
                _fail(error, 2)
    try:
        run_reference(
            ReferenceConfig(
                all_contigs=all_contigs,
                autosomal_contigs=(
                    tuple(autosomal_contig) if autosomal_contig else None
                ),
                bin_size=bin_size,
                fasta=fasta,
                max_batch_memory_mb=max_batch_memory_mb,
                min_cohort_size=min_cohort_size,
                min_sex_group_size=min_sex_group_size,
                nearest_bins=nearest_bins,
                pca_components=pca_components,
                prefix=prefix,
                qc_report=qc_report,
                samples=sample_paths,
                sex_contigs=tuple(sex_contig),
                sex_cutoff=sex_cutoff,
                sex_signal_contig=sex_signal_contig,
                threads=threads,
            )
        )
        cli_logger.debug("reference completed in %.3f s", perf_counter() - started)
    except ReferenceError as error:
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
    except ReferenceQCError as error:
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
            help="Sample Parquet input",
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
            help=(
                "Output path prefix for PREFIX_bins.bed, PREFIX_segments.bed, "
                "PREFIX_calls.bed, and PREFIX_plot.svg"
            ),
        ),
    ] = None,
    sex: Annotated[
        SexBranch,
        typer.Option(help="Sex branch used for gonosome models"),
    ] = SexBranch.autosomal,
    penalty: Annotated[
        float,
        typer.Option(min=0.0, help="Weighted-SSE segmentation penalty"),
    ] = 1.0,
    z_threshold: Annotated[
        float,
        typer.Option("--z-threshold", min=0.0, help="Absolute segment z call cutoff"),
    ] = 3.0,
    ratio_threshold: Annotated[
        float | None,
        typer.Option(
            "--ratio-threshold",
            min=0.0,
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
        typer.Option("--plot/--no-plot", help="Write a genome-aware SVG plot"),
    ] = True,
    loglevel: LogLevelOption = LogLevel.warning,
) -> None:
    """Predict copy-number events for one sample."""

    configure_logging(loglevel)
    cli_logger.debug(
        "starting predict: sample=%s, reference=%s, sex=%s",
        sample,
        reference,
        sex.value,
    )
    started = perf_counter()
    try:
        run_predict(
            PredictConfig(
                sample=sample,
                reference=reference,
                prefix=prefix if prefix is not None else sample.with_suffix(""),
                sex=sex.value,
                penalty=penalty,
                z_threshold=z_threshold,
                ratio_threshold=ratio_threshold,
                min_reference_bins=min_reference_bins,
                mask_repeats=mask_repeats,
                max_blacklist_gap=max_blacklist_gap,
                plot=plot,
            )
        )
        cli_logger.debug("predict completed in %.3f s", perf_counter() - started)
    except PredictError as error:
        _fail(error, 1)


def _fail(error: BaseException, code: int) -> NoReturn:
    typer.secho(str(error), err=True, fg="red")
    raise typer.Exit(code=code) from error


if __name__ == "__main__":
    app()
