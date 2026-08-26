import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from published_fixtures import CHR21_LENGTH, convert_published_samples
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.models import PredictConfig, ReadFilters, ReferenceConfig, parquet_path
from ccnv.predict import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    BinResult,
    BranchModel,
    PredictError,
    PredictionInput,
    SegmentResult,
    call_segment,
    load_inputs,
    normalize_branch,
    segment_and_call,
    write_outputs,
)
from ccnv.reference import BRANCH_AUTOSOMAL
from ccnv.reference import run as build_reference
from ccnv.schemas import (
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    metadata_for,
)


def _sample_table(*, include_chr2: bool = True) -> pa.Table:
    contigs = ["chr1", "chr2"] if include_chr2 else ["chr1"]
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths={"chr1": 100, "chr2": 80},
        values={
            "bin_size": 100,
            "source_identity": {"id": "test"},
            "reference_identity": {"id": "genome"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )
    return pa.table(
        {
            "contig": contigs,
            "start": pa.array([0] * len(contigs), type=pa.int64()),
            "end": pa.array([100, 80][: len(contigs)], type=pa.int64()),
            "count": pa.array([10, 10][: len(contigs)], type=pa.uint64()),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def _reference_table() -> pa.Table:
    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths={"chr1": 100, "chr2": 80},
        values={
            "model": {"branches": [BRANCH_AUTOSOMAL]},
            "sample_identities": [],
        },
    )
    return pa.table(
        {
            "branch": [BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
            "contig": ["chr1", "chr2"],
            "start": pa.array([0, 0], type=pa.int64()),
            "end": pa.array([100, 80], type=pa.int64()),
            "masked": [False, False],
            "pca_mean": pa.array([0.5, 0.5], type=pa.float64()),
            "pca_loadings": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array([[1], [0]], type=pa.list_(pa.int32())),
            "nearest_ref_distances": pa.array(
                [[1.0], [1.0]], type=pa.list_(pa.float64())
            ),
            "null_ratios": pa.array(
                [[-0.1, 0.1], [-0.1, 0.1]], type=pa.list_(pa.float64())
            ),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )


class TestNormalization(unittest.TestCase):
    def test_normalization_produces_aligned_finite_evidence(self) -> None:
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr2", "chr2"),
            starts=np.array([0, 100, 0, 100]),
            ends=np.array([100, 200, 100, 200]),
            counts=np.array([10.0, 10.0, 10.0, 10.0]),
            masked=np.zeros(4, dtype=np.bool_),
            pca_mean=np.full(4, 0.25),
            pca_loadings=np.zeros((1, 4)),
            nearest_indices=(
                np.array([2, 3]),
                np.array([2, 3]),
                np.array([0, 1]),
                np.array([0, 1]),
            ),
            nearest_distances=tuple(np.array([1.0, 1.2]) for _ in range(4)),
            null_ratios=tuple(np.array([-0.1, 0.0, 0.1]) for _ in range(4)),
        )
        config = PredictConfig(
            Path("sample"),
            Path("reference"),
            Path("output"),
            min_reference_bins=2,
        )

        result = normalize_branch(model, config)

        self.assertEqual(result.ratios.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.ratios)))
        self.assertTrue(np.all(np.isfinite(result.zscores)))
        np.testing.assert_allclose(result.ratios, 0.0)


class TestCalling(unittest.TestCase):
    def test_calls_symmetric_z_and_ratio_thresholds(self) -> None:
        self.assertEqual(
            call_segment(0.1, 4.0, z_threshold=3.0, ratio_threshold=None), "gain"
        )
        self.assertEqual(
            call_segment(-0.1, -4.0, z_threshold=3.0, ratio_threshold=None), "loss"
        )
        self.assertEqual(
            call_segment(0.3, 0.0, z_threshold=3.0, ratio_threshold=0.2), "gain"
        )
        self.assertIsNone(call_segment(0.1, 2.0, z_threshold=3.0, ratio_threshold=None))


class TestCoordinatesAndOutputs(unittest.TestCase):
    def test_bed_is_zero_based_half_open(self) -> None:
        bins = (
            BinResult(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1", "chr1"),
                starts=np.array([0, 95]),
                ends=np.array([95, 100]),
                ratios=np.array([0.5, np.nan]),
                zscores=np.array([4.0, np.nan]),
                weights=np.array([1.0, 0.0]),
                reference_sizes=np.array([10, 0]),
                masked=np.array([False, True]),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 95, 0.5, 4.0, 1.0, 1, "gain"),
            SegmentResult("chr1", 95, 100, 0.0, 0.0, 0.0, 1, None),
        )
        inputs = PredictionInput(
            _sample_table(),
            _reference_table(),
            {"chr1": 100, "chr2": 80},
            None,
            (BRANCH_AUTOSOMAL,),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                Path(tmpdir) / "result", bins, segments, inputs, plot=False
            )
            bins_rows = outputs.bins_bed.read_text().splitlines()
            segment_rows = outputs.segments_bed.read_text().splitlines()
            call_rows = outputs.calls_bed.read_text().splitlines()

        self.assertEqual(tuple(bins_rows[0].split("\t")), BINS_BED_HEADER)
        self.assertEqual(tuple(segment_rows[0].split("\t")), SEGMENTS_BED_HEADER)
        self.assertEqual(tuple(call_rows[0].split("\t")), CALLS_BED_HEADER)
        self.assertEqual(bins_rows[1].split("\t")[:3], ["chr1", "0", "95"])
        self.assertEqual(bins_rows[1].split("\t")[3], "chr1:0-95")
        self.assertEqual(
            bins_rows[2].split("\t")[:6],
            ["chr1", "95", "100", "chr1:95-100", "NaN", "NaN"],
        )
        self.assertEqual(bins_rows[2].split("\t")[8:], ["true", BRANCH_AUTOSOMAL])
        self.assertEqual(segment_rows[1].split("\t")[:3], ["chr1", "0", "95"])
        self.assertEqual(segment_rows[1].split("\t")[-1], "gain")
        self.assertEqual(segment_rows[2].split("\t")[-1], "")
        self.assertEqual(call_rows[1].split("\t")[:3], ["chr1", "0", "95"])
        self.assertEqual(len(call_rows), 2)

    def test_segmentation_clips_partial_contig_end(self) -> None:
        bins = (
            BinResult(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1", "chr1", "chr1"),
                starts=np.array([0, 100, 200]),
                ends=np.array([100, 200, 250]),
                ratios=np.array([0.0, 0.0, 0.0]),
                zscores=np.array([0.0, 0.0, 0.0]),
                weights=np.ones(3),
                reference_sizes=np.full(3, 10),
                masked=np.zeros(3, dtype=np.bool_),
            ),
        )

        segments = segment_and_call(
            bins,
            {"chr1": 250},
            penalty=1.0,
            z_threshold=3.0,
            ratio_threshold=None,
            max_blacklist_gap=2_000_000,
        )

        self.assertEqual((segments[-1].start, segments[-1].end), (0, 250))

    def test_missing_reference_contig_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.parquet"
            reference_path = directory / "reference.parquet"
            pq.write_table(_sample_table(include_chr2=False), sample_path)
            pq.write_table(_reference_table(), reference_path)

            with self.assertRaisesRegex(PredictError, "missing.*chr2"):
                load_inputs(sample_path, reference_path)


class TestPredictPublishedCrams(unittest.TestCase):
    def test_cli_predicts_published_sample_against_two_sample_reference(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_a, sample_b = convert_published_samples(root)
            reference_prefix = root / "reference"
            build_reference(
                ReferenceConfig(
                    samples=(sample_a, sample_b),
                    prefix=reference_prefix,
                    pca_components=1,
                    nearest_bins=16,
                    min_cohort_size=2,
                )
            )
            predict_prefix = root / "prediction"
            result = runner.invoke(
                app,
                [
                    "predict",
                    str(sample_a),
                    str(parquet_path(reference_prefix)),
                    "--prefix",
                    str(predict_prefix),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)

            bins_bed = Path(f"{predict_prefix}_bins.bed")
            segments_bed = Path(f"{predict_prefix}_segments.bed")
            calls_bed = Path(f"{predict_prefix}_calls.bed")
            plot_svg = Path(f"{predict_prefix}_plot.svg")
            bins_parquet = Path(f"{predict_prefix}_bins.parquet")
            segments_parquet = Path(f"{predict_prefix}_segments.parquet")
            for path in (
                bins_bed,
                segments_bed,
                calls_bed,
                plot_svg,
                bins_parquet,
                segments_parquet,
            ):
                self.assertTrue(path.is_file(), path)

            bin_lines = bins_bed.read_text().splitlines()
            self.assertEqual(tuple(bin_lines[0].split("\t")), BINS_BED_HEADER)
            self.assertEqual(len(bin_lines) - 1, 468)
            self.assertEqual(bin_lines[1].split("\t")[0], "chr21")
            self.assertEqual(int(bin_lines[1].split("\t")[1]), 0)
            self.assertEqual(int(bin_lines[-1].split("\t")[2]), CHR21_LENGTH)
            self.assertEqual(
                tuple(segments_bed.read_text().splitlines()[0].split("\t")),
                SEGMENTS_BED_HEADER,
            )
            self.assertEqual(
                tuple(calls_bed.read_text().splitlines()[0].split("\t")),
                CALLS_BED_HEADER,
            )
            bins_table = pq.read_table(bins_parquet)
            self.assertEqual(bins_table.num_rows, 468)
            self.assertEqual(set(bins_table.column("contig").to_pylist()), {"chr21"})
            self.assertGreater(plot_svg.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
