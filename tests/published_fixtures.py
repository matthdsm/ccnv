"""Published CRAM/FASTA fixtures downloaded on first functional-test run."""

import unittest
import urllib.error
import urllib.request
from pathlib import Path

from ccnv.convert import run as convert
from ccnv.models import ConvertConfig, parquet_path


def write_fai(fasta_path: Path) -> Path:
    """Generate a .fai index file for a FASTA in pure Python."""
    fai_path = Path(f"{fasta_path}.fai")
    records: list[str] = []
    with open(fasta_path, "rb") as handle:
        line = handle.readline()
        while line:
            if line.startswith(b">"):
                name = line[1:].split()[0].decode("ascii")
                offset = handle.tell()
                first_seq = handle.readline()
                line_blen = len(first_seq.rstrip(b"\r\n"))
                line_len = len(first_seq)
                seq_len = line_blen
                while True:
                    pos = handle.tell()
                    seq_line = handle.readline()
                    if not seq_line or seq_line.startswith(b">"):
                        handle.seek(pos)
                        break
                    seq_len += len(seq_line.rstrip(b"\r\n"))
                records.append(
                    f"{name}\t{seq_len}\t{offset}\t{line_blen}\t{line_len}\n"
                )
            line = handle.readline()
    fai_path.write_text("".join(records), encoding="ascii")
    return fai_path


CONVERT_FIXTURES = Path(__file__).parent / "fixtures" / "convert"
_DATASETS = "https://raw.githubusercontent.com/nf-cmgg/test-datasets/ccnv/"
PUBLISHED_CRAM_URL = f"{_DATASETS}data/genomics/homo_sapiens/illumina/cram/test.cram"
PUBLISHED_CRAM2_URL = f"{_DATASETS}data/genomics/homo_sapiens/illumina/cram/test2.cram"
PUBLISHED_FASTA_URL = (
    f"{_DATASETS}data/genomics/homo_sapiens/genome/seq/"
    "GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set_chr21.fna"
)
PUBLISHED_CRAM = CONVERT_FIXTURES / "test.cram"
PUBLISHED_CRAM2 = CONVERT_FIXTURES / "test2.cram"
PUBLISHED_FASTA = (
    CONVERT_FIXTURES / "GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set_chr21.fna"
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
    """Return the cached chr21 FASTA, indexing it when needed."""

    fasta = ensure_published_file(PUBLISHED_FASTA, PUBLISHED_FASTA_URL)
    if not Path(f"{fasta}.fai").is_file():
        write_fai(fasta)
    return fasta


def convert_published_samples(root: Path) -> tuple[Path, Path]:
    """Convert both published CRAMs into sample Parquet under ``root``."""

    fasta = ensure_published_fasta()
    paths: list[Path] = []
    for index, (cached, url) in enumerate(
        (
            (PUBLISHED_CRAM, PUBLISHED_CRAM_URL),
            (PUBLISHED_CRAM2, PUBLISHED_CRAM2_URL),
        ),
        start=1,
    ):
        cram = ensure_published_file(cached, url)
        prefix = root / f"sample{index}"
        convert(ConvertConfig(cram, prefix, PUBLISHED_BIN_SIZE, fasta=fasta))
        paths.append(parquet_path(prefix))
    return paths[0], paths[1]
