import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import polars_bio as pb
import pyarrow.parquet as pq
from published_fixtures import write_fai

from ccnv.convert import run as convert
from ccnv.models import (
    ConvertConfig,
    PredictConfig,
    ReferenceConfig,
    parquet_path,
)
from ccnv.predict import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    SEGMENTS_BED_HEADER,
)
from ccnv.predict import (
    run as predict,
)
from ccnv.reference import run as reference
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
            self.assertTrue(outputs.bins_bed.is_file())
            self.assertTrue(outputs.segments_bed.is_file())
            self.assertTrue(outputs.calls_bed.is_file())
            self.assertTrue(outputs.bins_parquet.is_file())
            self.assertTrue(outputs.segments_parquet.is_file())
            self.assertTrue(outputs.plot_svg and outputs.plot_svg.is_file())
            bins_header = outputs.bins_bed.read_text().splitlines()[0].split("\t")
            segments_header = (
                outputs.segments_bed.read_text().splitlines()[0].split("\t")
            )
            calls_header = outputs.calls_bed.read_text().splitlines()[0].split("\t")
            self.assertEqual(tuple(bins_header), BINS_BED_HEADER)
            self.assertEqual(tuple(segments_header), SEGMENTS_BED_HEADER)
            self.assertEqual(tuple(calls_header), CALLS_BED_HEADER)
            for line in outputs.bins_bed.read_text().splitlines()[1:]:
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
    write_fai(path)


def _write_alignment(path: Path, counts: dict[str, np.ndarray]) -> None:
    ref_seqs = [{"name": name, "length": length} for name, length in CONTIGS]
    (
        names,
        chroms,
        starts,
        ends,
        flags,
        mapqs,
        cigars,
        sequences,
        qualities,
        mate_chroms,
        mate_starts,
        tlen,
    ) = ([] for _ in range(12))
    read_index = 0
    for contig, length in CONTIGS:
        for bin_index, count in enumerate(counts[contig]):
            start = bin_index * BIN_SIZE
            for offset in range(int(count)):
                st = min(start + offset, length - 10)
                names.append(f"r{read_index}")
                chroms.append(contig)
                starts.append(st)
                ends.append(st + 10)
                flags.append(0)
                mapqs.append(60)
                cigars.append("10M")
                sequences.append("A" * 10)
                qualities.append("I" * 10)
                mate_chroms.append(None)
                mate_starts.append(None)
                tlen.append(0)
                read_index += 1

    df = pl.DataFrame(
        {
            "name": pl.Series(names, dtype=pl.String),
            "chrom": pl.Series(chroms, dtype=pl.String),
            "start": pl.Series(starts, dtype=pl.UInt32),
            "end": pl.Series(ends, dtype=pl.UInt32),
            "flags": pl.Series(flags, dtype=pl.UInt32),
            "cigar": pl.Series(cigars, dtype=pl.String),
            "mapping_quality": pl.Series(mapqs, dtype=pl.UInt32),
            "mate_chrom": pl.Series(mate_chroms, dtype=pl.String),
            "mate_start": pl.Series(mate_starts, dtype=pl.UInt32),
            "sequence": pl.Series(sequences, dtype=pl.String),
            "quality_scores": pl.Series(qualities, dtype=pl.String),
            "template_length": pl.Series(tlen, dtype=pl.Int32),
        }
    )
    meta_header = {
        "file_format_version": "1.6",
        "reference_sequences": json.dumps(ref_seqs),
    }
    pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, True)
    pb.set_source_metadata(df, format="bam", path=str(path), header=meta_header)
    pb.write_bam(df, str(path), sort_on_write=True)


def _bam_to_cram(bam_path: Path, cram_path: Path, fasta: Path) -> None:
    lf = pb.scan_bam(str(bam_path), use_zero_based=True)
    df = lf.collect()
    meta = pb.get_metadata(lf)
    meta_header = meta.get("header") or {}
    pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, True)
    pb.set_source_metadata(df, format="cram", path=str(cram_path), header=meta_header)
    pb.write_cram(df, str(cram_path), reference_path=str(fasta), sort_on_write=True)


if __name__ == "__main__":
    unittest.main()
