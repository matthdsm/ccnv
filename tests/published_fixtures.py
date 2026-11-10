"""Published CRAM/FASTA fixtures downloaded on first functional-test run."""

import unittest
import urllib.error
import urllib.request
from pathlib import Path

import pysam

from ccnv.extract import run as extract
from ccnv.models import ExtractConfig

EXTRACT_FIXTURES = Path(__file__).parent / "fixtures" / "extract"
_DATASETS = "https://raw.githubusercontent.com/nf-cmgg/test-datasets/wisecondorx/"
PUBLISHED_CRAM_URL = f"{_DATASETS}data/genomics/homo_sapiens/illumina/cram/test.cram"
PUBLISHED_CRAM2_URL = f"{_DATASETS}data/genomics/homo_sapiens/illumina/cram/test2.cram"
PUBLISHED_FASTA_URL = (
    f"{_DATASETS}data/genomics/homo_sapiens/genome/seq/"
    "GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set_chr21.fna"
)
PUBLISHED_CRAM = EXTRACT_FIXTURES / "test.cram"
PUBLISHED_CRAM2 = EXTRACT_FIXTURES / "test2.cram"
PUBLISHED_FASTA = (
    EXTRACT_FIXTURES / "GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set_chr21.fna"
)
CHR21_LENGTH = 46_709_983
PUBLISHED_BIN_SIZE = 100_000


def ensure_published_file(path: Path, url: str) -> Path:
    """Download ``url`` to ``path`` unless a non-empty file is already cached."""

    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
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
    tmp.replace(path)
    return path


def ensure_published_fasta() -> Path:
    """Return the cached chr21 FASTA, indexing it with pysam when needed."""

    fasta = ensure_published_file(PUBLISHED_FASTA, PUBLISHED_FASTA_URL)
    if not Path(f"{fasta}.fai").is_file():
        pysam.faidx(str(fasta))
    return fasta


def extract_published_samples(root: Path) -> tuple[Path, Path]:
    """Extract published CRAMs into sample Parquet under ``root``, reusing cache."""

    import shutil

    fasta = ensure_published_fasta()
    cache_dir = EXTRACT_FIXTURES / ".cache_extracted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (cached, url) in enumerate(
        (
            (PUBLISHED_CRAM, PUBLISHED_CRAM_URL),
            (PUBLISHED_CRAM2, PUBLISHED_CRAM2_URL),
        ),
        start=1,
    ):
        cram = ensure_published_file(cached, url)
        cached_sample = cache_dir / f"sample{index}.ccnv"
        valid = False
        if cached_sample.is_file() and cached_sample.stat().st_size > 0:
            try:
                from ccnv.io.parquet import read_sample_parquet
                from ccnv.models.exceptions import CcnvError

                read_sample_parquet(cached_sample)
                valid = True
            except CcnvError, OSError:
                valid = False
        if not valid:
            extract(
                ExtractConfig(
                    cram,
                    cache_dir / f"sample{index}",
                    PUBLISHED_BIN_SIZE,
                    fasta=fasta,
                )
            )
        dest = root / f"sample{index}.ccnv"
        if not dest.is_file() or dest.resolve() != cached_sample.resolve():
            root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached_sample, dest)
        paths.append(dest)
    return paths[0], paths[1]
