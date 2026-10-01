import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pysam

from ccnv.convert import run as convert
from ccnv.models import (
    ConvertConfig,
    PredictConfig,
    ReferenceConfig,
    parquet_path,
)
from ccnv.predict import run as predict
from ccnv.reference import run as reference
from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    SEGMENTS_BED_HEADER,
)
from ccnv.schemas import decode_metadata

CONTIGS = (
    ("chr2", 1_200),
    ("chr10", 1_150),
    ("linkage-X", 1_080),
    ("plasmid-Y", 930),
)
BIN_SIZE = 100


class TestInstalledPipelineContract(unittest.TestCase):
    def test_bam_and_cram_to_gain_loss_calls_on_nonhuman_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = root / "genome.fa"
            _write_fasta(fasta)

            sample_paths: list[Path] = []
            for sample_index in range(8):
                counts = _control_counts(sample_index)
                bam = root / f"control-{sample_index}.bam"
                _write_alignment(bam, counts)
                alignment = bam
                if sample_index == 0:
                    alignment = root / "control-0.cram"
                    _bam_to_cram(bam, alignment, fasta)
                prefix = root / f"control-{sample_index}"
                convert(
                    ConvertConfig(
                        alignment,
                        prefix,
                        BIN_SIZE,
                        fasta=fasta if alignment.suffix == ".cram" else None,
                        all_contigs=True,
                    )
                )
                sample_paths.append(parquet_path(prefix))

            target_counts = _control_counts(0)
            target_counts["chr10"][4:8] *= 2
            target_counts["linkage-X"][4:8] = np.maximum(
                1, target_counts["linkage-X"][4:8] // 4
            )
            target_bam = root / "target.bam"
            target_prefix = root / "target"
            target_sample = parquet_path(target_prefix)
            _write_alignment(target_bam, target_counts)
            convert(
                ConvertConfig(target_bam, target_prefix, BIN_SIZE, all_contigs=True)
            )

            reference_prefix = root / "reference"
            reference_path = parquet_path(reference_prefix)
            reference(
                ReferenceConfig(
                    tuple(sample_paths),
                    reference_prefix,
                    pca_components=2,
                    nearest_bins=12,
                    min_cohort_size=8,
                )
            )
            outputs = predict(
                PredictConfig(
                    target_sample,
                    reference_path,
                    root / "prediction",
                    penalty=0.05,
                    ratio_threshold=0.2,
                    min_reference_bins=3,
                    plot=True,
                )
            )

            sample = pq.read_table(target_sample)
            metadata = decode_metadata(sample.schema.metadata)
            self.assertEqual(
                metadata["reference_identity"]["contig_order"],
                [name for name, _ in CONTIGS],
            )
            self.assertIn("chr10", sample.column("contig").to_pylist())
            self.assertEqual(
                sample.filter(
                    np.asarray(sample.column("contig").to_pylist()) == "plasmid-Y"
                )
                .column("end")
                .to_pylist()[-1],
                930,
            )

            calls = {(item.contig, item.call) for item in outputs.segments if item.call}
            evidence = [
                (
                    result.contigs[index],
                    int(result.starts[index]),
                    float(result.ratios[index]),
                    float(result.zscores[index]),
                    bool(result.masked[index]),
                )
                for result in outputs.bins
                for index in range(len(result.contigs))
            ]
            self.assertIn(("chr10", "gain"), calls, evidence)
            self.assertIn(("linkage-X", "loss"), calls, evidence)
            self.assertTrue((root / "reference_qc.json").is_file())
            self.assertTrue(outputs.bins_bed.is_file())
            self.assertTrue(outputs.segments_bed.is_file())
            self.assertTrue(outputs.calls_bed.is_file())
            self.assertTrue(Path(str(outputs.bins_bed) + ".tbi").is_file())
            self.assertTrue(Path(str(outputs.segments_bed) + ".tbi").is_file())
            self.assertTrue(Path(str(outputs.calls_bed) + ".tbi").is_file())
            self.assertIsNotNone(outputs.bins_vcf)
            self.assertTrue(outputs.bins_vcf.is_file())
            self.assertTrue(Path(str(outputs.bins_vcf) + ".tbi").is_file())
            self.assertIsNotNone(outputs.segments_vcf)
            self.assertTrue(outputs.segments_vcf.is_file())
            self.assertTrue(Path(str(outputs.segments_vcf) + ".tbi").is_file())
            self.assertIsNotNone(outputs.calls_vcf)
            self.assertTrue(outputs.calls_vcf.is_file())
            self.assertTrue(Path(str(outputs.calls_vcf) + ".tbi").is_file())
            self.assertFalse((root / "prediction_bins.ccnv").exists())
            self.assertFalse((root / "prediction_segments.ccnv").exists())
            self.assertTrue(outputs.plot_svg and outputs.plot_svg.is_file())
            with gzip.open(outputs.bins_bed, "rt") as h:
                bins_lines = h.read().splitlines()
            bins_header = bins_lines[0].split("\t")
            with gzip.open(outputs.segments_bed, "rt") as h:
                segments_header = h.read().splitlines()[0].split("\t")
            with gzip.open(outputs.calls_bed, "rt") as h:
                calls_header = h.read().splitlines()[0].split("\t")
            self.assertEqual(tuple(bins_header), BINS_BED_HEADER)
            self.assertEqual(tuple(segments_header), SEGMENTS_BED_HEADER)
            self.assertEqual(tuple(calls_header), CALLS_BED_HEADER)
            for line in bins_lines[1:]:
                contig = line.split("\t")[0]
                self.assertFalse(contig.isdigit(), contig)

            for segment in outputs.segments:
                self.assertLessEqual(segment.end, dict(CONTIGS)[segment.contig])


def _control_counts(sample_index: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for contig_index, (contig, length) in enumerate(CONTIGS):
        bins = (length + BIN_SIZE - 1) // BIN_SIZE
        indexes = np.arange(bins)
        result[contig] = (
            20
            + (indexes * 3 + contig_index * 2 + sample_index) % 7
            + (sample_index % 3)
        ).astype(np.int64)
    return result


def _write_fasta(path: Path) -> None:
    with path.open("w", encoding="ascii") as handle:
        for name, length in CONTIGS:
            handle.write(f">{name}\n")
            handle.write("A" * length + "\n")
    pysam.faidx(str(path))


def _write_alignment(path: Path, counts: dict[str, np.ndarray]) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in CONTIGS],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as alignment:
        read_index = 0
        for reference_id, (contig, length) in enumerate(CONTIGS):
            for bin_index, count in enumerate(counts[contig]):
                start = bin_index * BIN_SIZE
                for offset in range(int(count)):
                    read = pysam.AlignedSegment()
                    read.query_name = f"r{read_index}"
                    read.query_sequence = "A" * 10
                    read.flag = 0
                    read.reference_id = reference_id
                    read.reference_start = min(start + offset, length - 10)
                    read.mapping_quality = 60
                    read.cigar = ((0, 10),)
                    read.query_qualities = pysam.qualitystring_to_array("I" * 10)
                    alignment.write(read)
                    read_index += 1
    pysam.index(str(path))


def _bam_to_cram(bam_path: Path, cram_path: Path, fasta: Path) -> None:
    with (
        pysam.AlignmentFile(str(bam_path), "rb") as bam,
        pysam.AlignmentFile(
            str(cram_path),
            "wc",
            header=bam.header,
            reference_filename=str(fasta),
        ) as cram,
    ):
        for read in bam:
            cram.write(read)
    pysam.index(str(cram_path))


if __name__ == "__main__":
    unittest.main()
