"""Core genomic and domain entities for ccnv."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
class Cytoband:
    """A cytogenetic band interval in 0-based half-open coordinates."""

    contig: str
    start: int
    end: int
    name: str
    stain: str


def normalize_chr(name: str) -> str:
    """Normalize a contig name by stripping any leading 'chr' prefix."""

    bare = name[3:] if name.lower().startswith("chr") else name
    return bare.upper() if bare.upper() in {"X", "Y"} else bare


def is_autosome_name(name: str) -> bool:
    """Return True if contig is a canonical numbered autosome (1..22)."""

    stripped = normalize_chr(name)
    return stripped.isdigit() and 1 <= int(stripped) <= 22


def resolve_contig_name(contig: str, available: Iterable[str]) -> str | None:
    """Resolve contig name against available keys with chr-prefix and case fallback."""

    avail_set = set(available)
    if contig in avail_set:
        return contig
    stripped = normalize_chr(contig)
    if stripped in avail_set:
        return stripped
    prefixed = f"chr{stripped}"
    if prefixed in avail_set:
        return prefixed
    lower_map = {normalize_chr(k).lower(): k for k in available}
    return lower_map.get(stripped.lower())


def is_canonical_contig(name: str) -> bool:
    """Report whether a contig is a canonical autosome or sex chromosome.

    Accepts UCSC (``chr1``, ``chrX``) and plain NCBI/Ensembl (``1``, ``X``) spellings.
    Mitochondrial contigs, ALT/decoy scaffolds, unplaced scaffolds, and EBV
    are not canonical.
    """

    return is_autosome_name(name) or normalize_chr(name) in {"X", "Y"}


def select_layout(
    names: Sequence[str],
    lengths: Sequence[int],
    *,
    all_contigs: bool,
) -> tuple[Contig, ...]:
    """Keep canonical chromosomes unless ``all_contigs`` is set.

    The result may be empty; callers decide how to report that. Header
    names and lengths must already be aligned.
    """

    if len(names) != len(lengths):
        raise ValueError("contig names and lengths must have equal length")
    return tuple(
        Contig(name, length)
        for name, length in zip(names, lengths, strict=True)
        if all_contigs or is_canonical_contig(name)
    )


__all__ = [
    "Contig",
    "Cytoband",
    "GenomeProfile",
    "SamplesheetEntry",
    "is_autosome_name",
    "is_canonical_contig",
    "normalize_chr",
    "resolve_contig_name",
    "select_layout",
]
