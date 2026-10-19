"""Typed domain objects shared by pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_BIN_SIZE: Final[int] = 5000


def parquet_path(prefix: Path) -> Path:
    """Canonical sample or reference Parquet path for a CLI prefix."""

    return Path(f"{prefix}.ccnv")


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
    samples: tuple[Path, ...]
    prefix: Path
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

    def __post_init__(self) -> None:
        if self.threads <= 0:
            raise ValueError("threads must be a positive integer")


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
