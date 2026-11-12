"""Core genomic and domain entities for ccnv."""

from __future__ import annotations

import re
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
        """Validate samplesheet entry attributes."""
        if not isinstance(self.filepath, Path):
            object.__setattr__(self, "filepath", Path(self.filepath))
        if self.sex is not None and self.sex not in ("M", "F"):
            raise ValueError(f"sex must be 'M' or 'F', got {self.sex!r}")
        if self.ff is not None and not (0.0 <= self.ff <= 1.0):
            raise ValueError(
                f"fetal fraction ff must be between 0.0 and 1.0, got {self.ff}"
            )


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
    """Return True if contig is a canonical numbered autosome (>= 1)."""

    stripped = normalize_chr(name)
    return stripped.isascii() and stripped.isdigit() and int(stripped) >= 1


@dataclass(frozen=True, slots=True)
class ContigLookup:
    """Pre-indexed contig resolver handling chr-prefix and case-insensitive matching."""

    exact: frozenset[str]
    lower_normalized: dict[str, str]

    @classmethod
    def from_names(cls, available: Iterable[str]) -> ContigLookup:
        """Build O(1) lookup tables from an iterable of available contig names."""
        names = tuple(dict.fromkeys(available))
        lower_map: dict[str, str] = {}
        for name in names:
            key = normalize_chr(name).lower()
            lower_map.setdefault(key, name)
        return cls(exact=frozenset(names), lower_normalized=lower_map)

    def resolve(self, contig: str) -> str | None:
        """Resolve a contig name against the indexed contig vocabulary."""
        if contig in self.exact:
            return contig
        stripped = normalize_chr(contig)
        if stripped in self.exact:
            return stripped
        prefixed = f"chr{stripped}"
        if prefixed in self.exact:
            return prefixed
        return self.lower_normalized.get(stripped.lower())


def is_canonical_contig(name: str) -> bool:
    """Report whether a contig is a canonical autosome or sex chromosome.

    Accepts UCSC (``chr1``, ``chrX``) and plain NCBI/Ensembl (``1``, ``X``) spellings.
    Mitochondrial contigs, ALT/decoy scaffolds, unplaced scaffolds, and EBV
    are not canonical.
    """

    return is_autosome_name(name) or normalize_chr(name) in {"X", "Y"}


def natural_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Return key for natural sort order treating embedded numbers as integers."""

    return tuple(
        (0, int(chunk)) if chunk.isdigit() else (1, chunk.casefold())
        for chunk in re.split(r"(\d+)", name)
        if chunk
    )


def select_layout(
    entries: Sequence[tuple[str, int]],
    *,
    all_contigs: bool = False,
) -> list[tuple[str, int]]:
    """Keep canonical chromosomes from ordered ``(name, length)`` header pairs unless ``all_contigs`` is set."""

    selected: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("header entries must be (name, length) pairs")
        contig_name, contig_len = entry
        if not isinstance(contig_name, str) or not contig_name:
            raise ValueError("contig name must not be empty")
        if (
            not isinstance(contig_len, int)
            or isinstance(contig_len, bool)
            or contig_len <= 0
        ):
            raise ValueError("contig length must be positive")
        if all_contigs or is_canonical_contig(contig_name):
            selected.append((contig_name, int(contig_len)))
    return selected


__all__ = [
    "ContigLookup",
    "Cytoband",
    "SamplesheetEntry",
    "is_autosome_name",
    "is_canonical_contig",
    "natural_sort_key",
    "normalize_chr",
    "select_layout",
]
