"""Configuration objects for ccnv pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from ccnv.models.domain import SamplesheetEntry

DEFAULT_BIN_SIZE: Final[int] = 5000
DEFAULT_MIN_MAPPABILITY: Final[float] = 0.5
AnalysisMode = Literal["germline", "prenatal"]


def parquet_path(prefix: Path) -> Path:
    """Canonical sample or reference Parquet path for a CLI prefix."""

    return Path(f"{prefix}.ccnv")


@dataclass(frozen=True, slots=True)
class ReadFilters:
    """Explicit per-read inclusion rules applied during extract."""

    min_mapping_quality: int = 1
    exclude_unmapped: bool = True
    exclude_secondary: bool = True
    exclude_supplementary: bool = True
    exclude_duplicates: bool = True
    require_proper_pair: bool = True

    def __post_init__(self) -> None:
        """Validate filter thresholds."""
        if self.min_mapping_quality < 0:
            raise ValueError("min_mapping_quality must be non-negative")


@dataclass(frozen=True, slots=True)
class ExtractConfig:
    alignment: Path
    prefix: Path
    bin_size: int = DEFAULT_BIN_SIZE
    fasta: Path | None = None
    filters: ReadFilters = ReadFilters()
    threads: int = 1
    all_contigs: bool = False
    min_frag_len: int = 30
    max_frag_len: int = 600

    def __post_init__(self) -> None:
        """Validate extraction configuration invariants."""
        if self.bin_size <= 0:
            raise ValueError("bin_size must be a positive integer")
        if self.threads <= 0:
            raise ValueError("threads must be a positive integer")
        if self.min_frag_len <= 0 or self.max_frag_len <= self.min_frag_len:
            raise ValueError("invalid min_frag_len / max_frag_len range")


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    samples: tuple[Path, ...] = ()
    prefix: Path = Path()
    autosomal_contigs: tuple[str, ...] | None = None
    sex_contigs: tuple[str, ...] | None = None
    sex_signal_contig: str | None = None
    sex_cutoff: float | None = None
    pca_components: int = 5
    nearest_bins: int = 300
    max_batch_memory_mb: int = 256
    min_cohort_size: int = 10
    min_sex_group_size: int = 5
    fasta: Path | None = None
    bin_size: int | None = None
    all_contigs: bool = False
    qc_report: bool = True
    threads: int = 1
    mappability: Path | None = None
    gc: Path | None = None
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    samplesheet_path: Path | None = None
    entries: tuple[SamplesheetEntry, ...] = ()
    mode: AnalysisMode = "germline"
    prenatal_bin_size: int = 100000
    seed: int = 42

    @property
    def input_paths(self) -> tuple[Path, ...]:
        """Backward-compatible alias for samples."""
        return self.samples

    def __post_init__(self) -> None:
        """Validate and normalize reference configuration invariants."""
        if isinstance(self.samples, (Path, str)):
            object.__setattr__(self, "prefix", Path(self.samples))
            object.__setattr__(self, "samples", ())

        if self.threads <= 0:
            raise ValueError("threads must be a positive integer")
        if self.prenatal_bin_size <= 0:
            raise ValueError("prenatal_bin_size must be a positive integer")
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )

        if not self.entries and self.samples:
            object.__setattr__(
                self,
                "entries",
                tuple(SamplesheetEntry(filepath=path) for path in self.samples),
            )
        elif not self.samples and self.entries:
            paths = tuple(entry.filepath for entry in self.entries)
            object.__setattr__(self, "samples", paths)


@dataclass(frozen=True, slots=True)
class ReferenceQCConfig:
    reference: Path
    prefix: Path | None = None


@dataclass(frozen=True, slots=True)
class PredictConfig:
    sample: Path
    reference: Path
    prefix: Path
    sex: Literal["autosomal", "female", "male"] = "autosomal"
    penalty: float = 1.0
    z_threshold: float = 3.0
    ratio_threshold: float | None = None
    min_reference_bins: int = 3
    mask_repeats: int = 3
    max_blacklist_gap: int = 2_000_000
    plot: bool = True
    vcf: bool = True
    cytobands: Path | None = None
    mappability: Path | None = None
    gc: Path | None = None
    fasta: Path | None = None
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    mode: AnalysisMode = "germline"
    min_fetal_fraction_warning: float = 0.04

    def __post_init__(self) -> None:
        """Validate prediction configuration invariants."""
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        if not (0.0 <= self.min_fetal_fraction_warning <= 1.0):
            raise ValueError("min_fetal_fraction_warning must be between 0.0 and 1.0")


__all__ = [
    "DEFAULT_BIN_SIZE",
    "DEFAULT_MIN_MAPPABILITY",
    "AnalysisMode",
    "ExtractConfig",
    "PredictConfig",
    "ReadFilters",
    "ReferenceConfig",
    "ReferenceQCConfig",
    "parquet_path",
]
