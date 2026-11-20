"""Testdata fixture resolver backed by ``copynumber-cnv-testdata``."""

from __future__ import annotations

import json
import os
import shutil
import unittest
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import yaml

from ccnv.extract import run as extract
from ccnv.models import ExtractConfig

EXTRACT_FIXTURES = Path(__file__).parent / "fixtures" / "extract"
_TESTDATA_URL = "https://raw.githubusercontent.com/matthdsm/ccnv-testdata/main/"
_SIBLING_TESTDATA_DIRS = (
    Path(__file__).resolve().parents[2] / "copynumber-cnv-testdata",
    Path(__file__).resolve().parents[2] / "ccnv-testdata",
)

PUBLISHED_BIN_SIZE = 5_000
PUBLISHED_CANONICAL_BINS = 12
PUBLISHED_CANONICAL_CONTIGS: dict[str, int] = {
    "chr1": 25_000,
    "chr2": 12_345,
    "chrX": 10_000,
    "chrY": 10_000,
}

_INDEX_SUFFIXES: dict[str, tuple[str, ...]] = {
    ".fa": (".fai",),
    ".fna": (".fai",),
    ".fasta": (".fai",),
    ".cram": (".crai",),
}


def _local_testdata_root() -> Path | None:
    env_dir = os.environ.get("CCNV_TESTDATA_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if candidate.is_dir():
            return candidate
    for sibling in _SIBLING_TESTDATA_DIRS:
        if sibling.is_dir():
            return sibling
    return None


def _download_file(dest: Path, url: str) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp")
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            tmp.open("wb") as handle,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as error:
        tmp.unlink(missing_ok=True)
        raise unittest.SkipTest(f"could not download {url}: {error}") from error
    tmp.replace(dest)
    return dest


def ensure_testdata_file(
    relpath: str,
    *,
    index_suffix: str | None = None,
) -> Path:
    """Resolve ``relpath`` from local ``copynumber-cnv-testdata`` or download and cache it."""
    local_root = _local_testdata_root()
    if local_root is not None:
        local_file = local_root / relpath
        if local_file.is_file():
            return local_file

    cached = EXTRACT_FIXTURES / "testdata" / relpath
    _download_file(cached, f"{_TESTDATA_URL}{relpath}")

    suffixes: tuple[str, ...] = (
        (index_suffix,)
        if index_suffix is not None
        else _INDEX_SUFFIXES.get(cached.suffix.lower(), ())
    )
    for sfx in suffixes:
        _download_file(
            Path(f"{cached}{sfx}"),
            f"{_TESTDATA_URL}{relpath}{sfx}",
        )
    return cached


def ensure_testdata_samplesheet(relpath: str) -> Path:
    """Resolve a testdata YAML/JSON samplesheet and ensure every relative sample file."""
    sheet_path = ensure_testdata_file(relpath)
    raw = (
        json.loads(sheet_path.read_text(encoding="utf-8"))
        if sheet_path.suffix.lower() == ".json"
        else yaml.safe_load(sheet_path.read_text(encoding="utf-8"))
    )
    if not isinstance(raw, list):
        raise TypeError(f"expected list in samplesheet {relpath}")
    parent = PurePosixPath(relpath).parent
    for entry in raw:
        if isinstance(entry, Mapping):
            fp = entry.get("filepath")
            if isinstance(fp, str) and fp:
                rel_sample = str(parent / fp) if str(parent) != "." else fp
                ensure_testdata_file(rel_sample)
    return sheet_path


def load_testdata_truth(relpath: str) -> dict[str, object]:
    """Load and decode a JSON ground-truth manifest from ``copynumber-cnv-testdata``."""
    path = ensure_testdata_file(relpath)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"expected JSON object in {relpath}")
    return loaded


def require_dict(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    """Extract a required nested JSON object from ``mapping`` with strict type narrowing."""
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict for key {key!r}, got {type(value).__name__}")
    return value


def require_float(mapping: Mapping[str, object], key: str) -> float:
    """Extract a required numeric JSON field as ``float`` with strict type narrowing."""
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected number for key {key!r}, got {type(value).__name__}")
    return float(value)


def optional_str_list(mapping: Mapping[str, object], key: str) -> list[str]:
    """Extract an optional JSON list of strings with strict type narrowing."""
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"expected list[str] for key {key!r}, got {value!r}")
    return list(value)


def ensure_published_fasta() -> Path:
    """Return the synthetic reference FASTA (with ``.fai`` index)."""
    return ensure_testdata_file("extract/sample_data/synthetic_ref.fa")


def ensure_published_cram() -> Path:
    """Return the synthetic paired-end CRAM (with ``.crai`` index)."""
    return ensure_testdata_file("extract/sample_data/paired_end.cram")


def ensure_published_bam() -> Path:
    """Return the synthetic paired-end BAM (with ``.bai`` index)."""
    return ensure_testdata_file(
        "extract/sample_data/paired_end.bam", index_suffix=".bai"
    )


def extract_published_samples(root: Path) -> tuple[Path, Path]:
    """Extract the synthetic paired-end CRAM and BAM into sample Parquets under ``root``."""
    fasta = ensure_published_fasta()
    cram = ensure_published_cram()
    bam = ensure_published_bam()
    cache_dir = EXTRACT_FIXTURES / ".cache_extracted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, aln, ref_fa in (
        ("synthetic_cram", cram, fasta),
        ("synthetic_bam", bam, None),
    ):
        cached_sample = cache_dir / f"{name}.ccnv"
        valid = False
        if cached_sample.is_file() and cached_sample.stat().st_size > 0:
            try:
                from ccnv.io.parquet import read_sample_buffers
                from ccnv.models.exceptions import CcnvError

                bufs = read_sample_buffers(cached_sample)
                valid = (
                    bufs.meta.bin_size == PUBLISHED_BIN_SIZE
                    and bufs.meta.contig_lengths == PUBLISHED_CANONICAL_CONTIGS
                )
            except (CcnvError, OSError):
                valid = False
        if not valid:
            extract(
                ExtractConfig(
                    aln,
                    cache_dir / name,
                    PUBLISHED_BIN_SIZE,
                    fasta=ref_fa,
                )
            )
        dest = root / f"{name}.ccnv"
        if not dest.is_file() or dest.resolve() != cached_sample.resolve():
            root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_sample, dest)
        paths.append(dest)
    return paths[0], paths[1]
