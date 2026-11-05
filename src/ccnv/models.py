"""Typed domain objects shared by pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_BIN_SIZE: Final[int] = 5000
DEFAULT_MIN_MAPPABILITY: Final[float] = 0.5
AnalysisMode = Literal["germline", "prenatal"]


def parquet_path(prefix: Path) -> Path:
    """Canonical sample or reference Parquet path for a CLI prefix."""

    return Path(f"{prefix}.ccnv")


@dataclass(frozen=True, slots=True)
class SamplesheetEntry:
    """One entry in a cohort reference samplesheet."""

    filepath: Path
    sex: Literal["M", "F"] | None = None
    ff: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.filepath, Path):
            object.__setattr__(self, "filepath", Path(self.filepath))
        if self.sex is not None and self.sex not in ("M", "F"):
            raise ValueError(f"sex must be 'M' or 'F', got {self.sex!r}")
        if self.ff is not None and not (0.0 <= self.ff <= 1.0):
            raise ValueError(
                f"fetal fraction ff must be between 0.0 and 1.0, got {self.ff}"
            )


@dataclass(frozen=True, slots=True)
class Contig:
    """A contig exactly as declared by an alignment header."""

    name: str
    length: int
    role: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("contig name must not be empty")
        if self.length <= 0:
            raise ValueError("contig length must be positive")


@dataclass(frozen=True, slots=True)
class GenomeProfile:
    """Ordered contigs and optional biological roles for one genome."""

    contigs: tuple[Contig, ...]

    def __post_init__(self) -> None:
        names = [contig.name for contig in self.contigs]
        if not names:
            raise ValueError("genome profile must contain at least one contig")
        if len(names) != len(set(names)):
            raise ValueError("contig names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(contig.name for contig in self.contigs)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(contig.length for contig in self.contigs)

    def length_map(self) -> dict[str, int]:
        return {contig.name: contig.length for contig in self.contigs}


@dataclass(frozen=True, slots=True)
class ReadFilters:
    """Explicit per-read inclusion rules applied during convert."""

    min_mapping_quality: int = 1
    exclude_unmapped: bool = True
    exclude_secondary: bool = True
    exclude_supplementary: bool = True
    exclude_duplicates: bool = True
    require_proper_pair: bool = True

    def __post_init__(self) -> None:
        if self.min_mapping_quality < 0:
            raise ValueError("min_mapping_quality must be non-negative")


@dataclass(frozen=True, slots=True)
class ConvertConfig:
    alignment: Path
    prefix: Path
    bin_size: int = DEFAULT_BIN_SIZE
    fasta: Path | None = None
    filters: ReadFilters = ReadFilters()
    threads: int = 1
    all_contigs: bool = False

    def __post_init__(self) -> None:
        if self.bin_size <= 0:
            raise ValueError("bin_size must be a positive integer")
        if self.threads <= 0:
            raise ValueError("threads must be a positive integer")


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
    input_paths: tuple[Path, ...] = ()
    mode: AnalysisMode = "germline"
    prenatal_bin_size: int = 100000

    def __post_init__(self) -> None:
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

        # Keep samples, input_paths, and entries synchronized for backward compatibility
        if not self.samples and self.input_paths:
            object.__setattr__(self, "samples", self.input_paths)
        elif not self.input_paths and self.samples:
            object.__setattr__(self, "input_paths", self.samples)

        if not self.entries and self.samples:
            object.__setattr__(
                self,
                "entries",
                tuple(SamplesheetEntry(filepath=path) for path in self.samples),
            )
        elif not self.samples and self.entries:
            paths = tuple(entry.filepath for entry in self.entries)
            object.__setattr__(self, "samples", paths)
            object.__setattr__(self, "input_paths", paths)


@dataclass(frozen=True, slots=True)
class ReferenceQCConfig:
    reference: Path
    prefix: Path | None = None


@dataclass(frozen=True, slots=True)
class Cytoband:
    """A cytogenetic band interval in 0-based half-open coordinates."""

    contig: str
    start: int
    end: int
    name: str
    stain: str


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
    min_mappability: float = DEFAULT_MIN_MAPPABILITY
    mode: AnalysisMode = "germline"
    min_fetal_fraction_warning: float = 0.04

    def __post_init__(self) -> None:
        if self.mode not in ("germline", "prenatal"):
            raise ValueError(
                f"mode must be 'germline' or 'prenatal', got {self.mode!r}"
            )
        if not (0.0 <= self.min_fetal_fraction_warning <= 1.0):
            raise ValueError("min_fetal_fraction_warning must be between 0.0 and 1.0")
