import gzip
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from published_fixtures import CHR21_LENGTH, extract_published_samples
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.models import PredictConfig, ReadFilters, ReferenceConfig, parquet_path
from ccnv.models.results import BinResult, SegmentResult
from ccnv.models.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    PRENATAL_METADATA_KEY,
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    PrenatalModelMetadata,
    decode_metadata,
    metadata_for,
)
from ccnv.predict import (
    BranchModel,
    PredictError,
    PredictionInput,
    call_segment,
    load_inputs,
    normalize_branch,
    segment_and_call,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as build_reference
from ccnv.report import (
    BINS_BED_HEADER,
    CALLS_BED_HEADER,
    SEGMENTS_BED_HEADER,
    write_outputs,
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
            "model": {"branches": [BRANCH_AUTOSOMAL], "bin_size": 100},
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
            "gc": pa.array([float("nan"), float("nan")], type=pa.float32()),
            "mappability": pa.array([float("nan"), float("nan")], type=pa.float32()),
            "pca_mean": pa.array([0.5, 0.5], type=pa.float64()),
            "pca_loadings": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array([[1], [0]], type=pa.list_(pa.int32())),
            "mean_distance": pa.array([1.0, 1.0], type=pa.float32()),
            "null_mean": pa.array([0.0, 0.0], type=pa.float64()),
            "null_std": pa.array([0.141421356, 0.141421356], type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )


def _prenatal_reference_table() -> pa.Table:
    prenatal_meta = PrenatalModelMetadata(
        prenatal_bin_size=100_000,
        yfrac_threshold=0.005,
        ffy_intercept=0.0,
        ffy_slope=1.0,
        mlp_w1=[[0.1, -0.1]],
        mlp_b1=[0.0, 0.0],
        mlp_w2=[[0.5], [-0.5]],
        mlp_b2=[0.0],
        mlp_skew_intercept=0.0,
        mlp_skew_slope=1.0,
        olm_weights=[0.1],
        olm_bias=0.0,
        olm_skew_intercept=0.0,
        olm_skew_slope=1.0,
        ffx_intercept=1.0,
        ffx_slope=-0.5,
        female_chrx_ratio_mean=1.0,
        female_chrx_ratio_std=0.01,
    )
    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths={
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 300_000,
        },
        values={
            "model": {
                "branches": [
                    BRANCH_AUTOSOMAL,
                    BRANCH_FEMALE_SEX,
                    BRANCH_PRENATAL_AUTOSOMAL,
                    BRANCH_PRENATAL_FFY,
                ],
                "bin_size": 100_000,
                "mode": "prenatal",
                "prenatal_bin_size": 100_000,
                "sex_signal_contig": "chrY",
                "sex_cutoff": 0.005,
                "sex_inference": {"cutoff": 0.005},
            },
            "sample_identities": [],
        },
    )
    metadata[PRENATAL_METADATA_KEY] = prenatal_meta.encode()

    branches: list[str] = []
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    masked: list[bool] = []
    pca_mean: list[float] = []
    pca_loadings: list[list[float]] = []
    nearest_ref: list[list[int]] = []
    mean_dist: list[float] = []
    null_mean: list[float] = []
    null_std: list[float] = []

    for c in ["chr1", "chr2"]:
        for s, e in [(0, 100_000), (100_000, 200_000)]:
            branches.append(BRANCH_AUTOSOMAL)
            contigs.append(c)
            starts.append(s)
            ends.append(e)
            masked.append(False)
            pca_mean.append(0.25)
            pca_loadings.append([0.0])
            nearest_ref.append([0])
            mean_dist.append(1.0)
            null_mean.append(0.0)
            null_std.append(0.1)

    for s, e in [(0, 100_000), (100_000, 200_000)]:
        branches.append(BRANCH_FEMALE_SEX)
        contigs.append("chrX")
        starts.append(s)
        ends.append(e)
        masked.append(False)
        pca_mean.append(0.5)
        pca_loadings.append([0.0])
        nearest_ref.append([0])
        mean_dist.append(1.0)
        null_mean.append(0.0)
        null_std.append(0.1)

    for c in ["chr1", "chr2"]:
        for s, e in [(0, 100_000), (100_000, 200_000)]:
            branches.append(BRANCH_PRENATAL_AUTOSOMAL)
            contigs.append(c)
            starts.append(s)
            ends.append(e)
            masked.append(False)
            pca_mean.append(0.0)
            pca_loadings.append([0.5])
            nearest_ref.append([0])
            mean_dist.append(1.0)
            null_mean.append(0.0)
            null_std.append(0.1)

    for s, e in [(0, 100_000), (100_000, 200_000), (200_000, 300_000)]:
        branches.append(BRANCH_PRENATAL_FFY)
        contigs.append("chrY")
        starts.append(s)
        ends.append(e)
        masked.append(False)
        pca_mean.append(0.33)
        pca_loadings.append([0.0])
        nearest_ref.append([0])
        mean_dist.append(1.0)
        null_mean.append(0.0)
        null_std.append(0.1)

    n_rows = len(branches)
    return pa.table(
        {
            "branch": branches,
            "contig": contigs,
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "masked": masked,
            "gc": pa.array([float("nan")] * n_rows, type=pa.float32()),
            "mappability": pa.array([float("nan")] * n_rows, type=pa.float32()),
            "pca_mean": pa.array(pca_mean, type=pa.float64()),
            "pca_loadings": pa.array(pca_loadings, type=pa.list_(pa.float64())),
            "nearest_ref_indices": pa.array(nearest_ref, type=pa.list_(pa.int32())),
            "mean_distance": pa.array(mean_dist, type=pa.float32()),
            "null_mean": pa.array(null_mean, type=pa.float64()),
            "null_std": pa.array(null_std, type=pa.float64()),
        },
        schema=REFERENCE_SCHEMA.with_metadata(metadata),
    )


def _prenatal_sample_table() -> pa.Table:
    contig_lengths = {
        "chr1": 200_000,
        "chr2": 200_000,
        "chrX": 200_000,
        "chrY": 300_000,
    }
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": 100_000,
            "source_identity": {"path": "test_sample.ccnv"},
            "reference_identity": {"id": "genome"},
            "qc": {
                "effective_reads_examined": 500,
                "effective_reads_kept": 400,
                "autosomal_reads_kept": 380,
                "sex_reads_kept": 20,
            },
            "filters": asdict(ReadFilters()),
        },
    )
    contigs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    counts: list[int] = []
    for c, length in contig_lengths.items():
        for s in range(0, length, 100_000):
            contigs.append(c)
            starts.append(s)
            ends.append(s + 100_000)
            cnt = 5 if c == "chrY" else 100
            counts.append(cnt)

    return pa.table(
        {
            "contig": contigs,
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "count": pa.array(counts, type=pa.uint64()),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
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
            mean_distance=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            null_mean=np.zeros(4, dtype=np.float64),
            null_std=np.full(4, 0.1, dtype=np.float64),
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

    def test_project_coverage_lstsq_reconstruction_with_masked_and_empty_active(
        self,
    ) -> None:
        from ccnv.predict import project_coverage

        # 4 bins, 1 component: loading = [0.5, 0.5, 0.5, 0.5]
        # Mean = [0.25, 0.25, 0.25, 0.25]
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr2", "chr2"),
            starts=np.array([0, 100, 0, 100], dtype=np.int64),
            ends=np.array([100, 200, 100, 200], dtype=np.int64),
            counts=np.array([20.0, 30.0, 20.0, 30.0]),
            masked=np.zeros(4, dtype=np.bool_),
            pca_mean=np.full(4, 0.25),
            pca_loadings=np.full((1, 4), 0.5),
            nearest_indices=(
                np.array([2, 3]),
                np.array([2, 3]),
                np.array([0, 1]),
                np.array([0, 1]),
            ),
            mean_distance=np.ones(4, dtype=np.float32),
            null_mean=np.zeros(4, dtype=np.float64),
            null_std=np.full(4, 0.1, dtype=np.float64),
        )

        # 1. Mask bin 0 and 1 -> active_sub is [False, False, True, True]
        n_mask = np.array([True, True, False, False])
        proj = project_coverage(model, n_mask_override=n_mask)
        # Usable count sum = 20.0 + 30.0 = 50.0 -> selected is [0.4, 0.6]
        # sub_loadings.T is [[0.5], [0.5]]; lstsq solves c = 0.5
        # recon = [0.5, 0.5, 0.5, 0.5]
        self.assertTrue(np.all(proj.usable == np.array([False, False, True, True])))
        self.assertAlmostEqual(proj.corrected[2], 0.4 / 0.5)
        self.assertAlmostEqual(proj.corrected[3], 0.6 / 0.5)

        # 2. All bins masked -> raises PredictError (zero usable count)
        all_masked = np.ones(4, dtype=np.bool_)
        with self.assertRaises(PredictError):
            project_coverage(model, n_mask_override=all_masked)

        # 3. One active bin -> n_active (1) <= rank (1) -> fallback to pca_mean copy
        one_active_mask = np.array([True, True, True, False])
        proj_one = project_coverage(model, n_mask_override=one_active_mask)
        self.assertTrue(
            np.all(proj_one.usable == np.array([False, False, False, True]))
        )
        # Active bin 3 count 30.0, usable_total=30.0 -> normalized = 1.0
        # n_active=1 <= rank=1 -> reconstruction is pca_mean[3] = 0.25
        self.assertAlmostEqual(proj_one.corrected[3], 1.0 / 0.25)

    def test_project_coverage_huber_resists_trisomy_outlier_chromosome(self) -> None:
        from ccnv.predict import project_coverage

        # 20 bins: 16 diploid bins follow loading, 4 trisomic bins have +50% jump
        loadings = np.linspace(-1.0, 1.0, 20).reshape(1, 20)
        pca_mean = np.full(20, 0.05)
        true_c = 0.02
        counts = (pca_mean + true_c * loadings.ravel()) * 1000.0
        counts[16:] *= 1.8  # Trisomy / amplification on last 4 bins

        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1",) * 16 + ("chr21",) * 4,
            starts=np.arange(20, dtype=np.int64) * 100,
            ends=np.arange(1, 21, dtype=np.int64) * 100,
            counts=counts,
            masked=np.zeros(20, dtype=np.bool_),
            pca_mean=pca_mean,
            pca_loadings=loadings,
            nearest_indices=tuple(np.arange(16, dtype=np.int32) for _ in range(20)),
            mean_distance=np.ones(20, dtype=np.float32),
            null_mean=np.zeros(20, dtype=np.float64),
            null_std=np.full(20, 0.05, dtype=np.float64),
        )
        proj = project_coverage(model)
        # Diploid bins (0..15) should have near-flat corrected ratios (~1.0), not tilted by chr21
        self.assertLess(float(np.std(proj.corrected[:16])), 0.05)
        self.assertGreater(float(np.mean(proj.corrected[16:])), 1.4)


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

    def test_segment_and_call_stouffer_zscore_accumulates_evidence_and_corrects_autocorr(
        self,
    ) -> None:
        # 25 consecutive bins with Z_i = +1.8 (white noise / zero lag-1 autocorr via alternating jitter)
        z = np.full(25, 1.8, dtype=np.float64) + np.tile([0.05, -0.05], 13)[:25]
        bins = (
            BinResult(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1",) * 25,
                starts=np.arange(25, dtype=np.int64) * 100,
                ends=np.arange(1, 26, dtype=np.int64) * 100,
                ratios=np.full(25, 0.35, dtype=np.float64),
                zscores=z,
                weights=np.ones(25, dtype=np.float64),
                reference_sizes=np.full(25, 10, dtype=np.int64),
                masked=np.zeros(25, dtype=np.bool_),
            ),
        )
        segs = segment_and_call(
            bins,
            {"chr1": 2500},
            penalty=5.0,
            z_threshold=3.0,
            ratio_threshold=None,
        )
        self.assertEqual(len(segs), 1)
        # Mean Z would be ~1.8 (<3.0, missed); Stouffer Z is ~1.8 * sqrt(25) = 9.0 (>3.0, called gain)
        self.assertGreater(segs[0].zscore, 8.0)
        self.assertEqual(segs[0].call, "gain")

    def test_segment_and_call_preserves_zero_usable_bins_segment(self) -> None:
        bins = (
            BinResult(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1",) * 6,
                starts=np.arange(6, dtype=np.int64) * 100,
                ends=np.arange(1, 7, dtype=np.int64) * 100,
                ratios=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
                zscores=np.array(
                    [3.5, 3.5, 3.5, np.nan, np.nan, np.nan], dtype=np.float64
                ),
                weights=np.ones(6, dtype=np.float64),
                reference_sizes=np.full(6, 10, dtype=np.int64),
                masked=np.zeros(6, dtype=np.bool_),
            ),
        )
        segs = segment_and_call(
            bins,
            {"chr1": 600},
            penalty=0.1,
            z_threshold=3.0,
            ratio_threshold=None,
        )
        self.assertEqual(len(segs), 2)
        matched = False
        for s in segs:
            if s.start == 300 and s.end == 600:
                self.assertEqual(s.zscore, 0.0)
                self.assertIsNone(s.call)
                matched = True
        self.assertTrue(matched)


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
                Path(tmpdir) / "result",
                bins,
                segments,
                inputs.contig_lengths,
                plot=False,
            )
            with gzip.open(outputs.bins_bed, "rt") as h:
                bins_rows = h.read().splitlines()
            with gzip.open(outputs.segments_bed, "rt") as h:
                segment_rows = h.read().splitlines()
            with gzip.open(outputs.calls_bed, "rt") as h:
                call_rows = h.read().splitlines()

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
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            pq.write_table(_sample_table(include_chr2=False), sample_path)
            pq.write_table(_reference_table(), reference_path)

            with self.assertRaisesRegex(PredictError, "missing.*chr2"):
                load_inputs(sample_path, reference_path)

    def test_load_inputs_rebins_sample_without_modifying_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"

            sample_metadata = metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 100, "chr2": 80},
                values={
                    "bin_size": 50,
                    "source_identity": {"id": "test"},
                    "reference_identity": {"id": "genome"},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
            sample_50 = pa.table(
                {
                    "contig": ["chr1", "chr1", "chr2", "chr2"],
                    "start": pa.array([0, 50, 0, 50], type=pa.int64()),
                    "end": pa.array([50, 100, 50, 80], type=pa.int64()),
                    "count": pa.array([5, 5, 6, 4], type=pa.uint64()),
                },
                schema=SAMPLE_SCHEMA.with_metadata(sample_metadata),
            )
            pq.write_table(sample_50, sample_path)
            ref_table = _reference_table()
            pq.write_table(ref_table, reference_path)

            ref_bytes_before = reference_path.read_bytes()
            ref_mtime_before = reference_path.stat().st_mtime_ns

            inputs = load_inputs(sample_path, reference_path)

            self.assertEqual(inputs.sample.num_rows, 2)
            self.assertEqual(
                decode_metadata(inputs.sample.schema.metadata)["bin_size"], 100
            )
            np.testing.assert_allclose(
                inputs.sample.column("count").to_pylist(), [10.0, 10.0]
            )
            self.assertEqual(reference_path.read_bytes(), ref_bytes_before)
            self.assertEqual(reference_path.stat().st_mtime_ns, ref_mtime_before)

    def test_load_inputs_rejects_contig_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"

            sample_metadata = metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 150, "chr2": 80},
                values={
                    "bin_size": 100,
                    "source_identity": {"id": "test"},
                    "reference_identity": {"id": "genome"},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
            sample_mismatch = pa.table(
                {
                    "contig": ["chr1", "chr1", "chr2"],
                    "start": pa.array([0, 100, 0], type=pa.int64()),
                    "end": pa.array([100, 150, 80], type=pa.int64()),
                    "count": pa.array([10, 5, 10], type=pa.uint64()),
                },
                schema=SAMPLE_SCHEMA.with_metadata(sample_metadata),
            )
            pq.write_table(sample_mismatch, sample_path)
            pq.write_table(_reference_table(), reference_path)

            with self.assertRaisesRegex(PredictError, "contig length mismatch"):
                load_inputs(sample_path, reference_path)

    def test_build_branch_model_rejects_mismatched_sample_coordinates(self) -> None:
        from ccnv.predict import build_branch_model

        ref = _reference_table()
        bad_sample = pa.table(
            {
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([100], type=pa.int64()),
                "count": pa.array([10], type=pa.uint64()),
            },
            schema=SAMPLE_SCHEMA.with_metadata(
                metadata_for(
                    ArtifactKind.SAMPLE,
                    contig_lengths={"chr1": 100},
                    values={
                        "bin_size": 100,
                        "source_identity": {"id": "test"},
                        "reference_identity": {"id": "genome"},
                        "qc": {},
                        "filters": asdict(ReadFilters()),
                    },
                )
            ),
        )
        inputs = PredictionInput(
            bad_sample,
            ref,
            {"chr1": 100, "chr2": 80},
            None,
            (BRANCH_AUTOSOMAL,),
        )
        with self.assertRaises(PredictError):
            build_branch_model(inputs, BRANCH_AUTOSOMAL)


class TestModeGuardrails(unittest.TestCase):
    def test_prenatal_mode_with_germline_ref_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            pq.write_table(_sample_table(), sample_path)
            pq.write_table(_reference_table(), reference_path)

            expected_msg = (
                "reference was built in germline mode and lacks prenatal models; "
                "rebuild reference with --mode prenatal "
                "or run predict with --mode germline"
            )
            with self.assertRaisesRegex(PredictError, expected_msg):
                load_inputs(sample_path, reference_path, mode="prenatal")

    def test_germline_mode_with_prenatal_ref_warns_and_filters_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            pq.write_table(_prenatal_sample_table(), sample_path)
            pq.write_table(_prenatal_reference_table(), reference_path)

            with self.assertLogs("ccnv.predict", level="WARNING") as cm:
                inputs = load_inputs(sample_path, reference_path, mode="germline")

            has_warn = any(
                "prenatal models are ignored in germline mode" in log
                for log in cm.output
            )
            self.assertTrue(has_warn)
            self.assertIn(BRANCH_AUTOSOMAL, inputs.branches)
            self.assertNotIn(BRANCH_PRENATAL_AUTOSOMAL, inputs.branches)
            self.assertNotIn(BRANCH_PRENATAL_FFY, inputs.branches)


class TestPrenatalPredictPipeline(unittest.TestCase):
    def test_predict_prenatal_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            prefix = directory / "prenatal_out"
            pq.write_table(_prenatal_sample_table(), sample_path)
            pq.write_table(_prenatal_reference_table(), reference_path)

            config = PredictConfig(
                sample=sample_path,
                reference=reference_path,
                prefix=prefix,
                mode="prenatal",
                min_reference_bins=1,
            )
            outputs = run_predict(config)

            self.assertIsNotNone(outputs.prenatal_result)
            self.assertIsNotNone(outputs.qc_json_path)
            self.assertTrue(outputs.qc_json_path.is_file())

            # Read and verify QC JSON
            with open(outputs.qc_json_path, encoding="utf-8") as f:
                qc_data = json.load(f)

            self.assertEqual(qc_data["mode"], "prenatal")
            self.assertIsNotNone(qc_data["prenatal"])
            self.assertEqual(qc_data["prenatal"]["fetal_sex"], "M")
            self.assertGreater(qc_data["prenatal"]["active_ff"], 0.0)
            self.assertIn("read_metrics", qc_data)
            self.assertIn("cnv_summary", qc_data)

            # Segments have prenatal annotations
            for seg in outputs.segments:
                self.assertIsNotNone(seg.expected_ratio)
                self.assertIsNotNone(seg.implied_ff)
                self.assertIsInstance(seg.discordant, bool)


class TestPredictPublishedCrams(unittest.TestCase):
    def test_cli_predicts_published_sample_against_two_sample_reference(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_a, sample_b = extract_published_samples(root)
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

            bins_bed = Path(f"{predict_prefix}_bins.bed.gz")
            segments_bed = Path(f"{predict_prefix}_segments.bed.gz")
            calls_bed = Path(f"{predict_prefix}_calls.bed.gz")
            bins_vcf = Path(f"{predict_prefix}_bins.vcf.gz")
            segments_vcf = Path(f"{predict_prefix}_segments.vcf.gz")
            calls_vcf = Path(f"{predict_prefix}_calls.vcf.gz")
            plot_html = Path(f"{predict_prefix}_plot.html")
            qc_json = Path(f"{predict_prefix}_qc.json")
            for path in (
                bins_bed,
                segments_bed,
                calls_bed,
                Path(str(bins_bed) + ".tbi"),
                Path(str(segments_bed) + ".tbi"),
                Path(str(calls_bed) + ".tbi"),
                bins_vcf,
                segments_vcf,
                calls_vcf,
                Path(str(bins_vcf) + ".tbi"),
                Path(str(segments_vcf) + ".tbi"),
                Path(str(calls_vcf) + ".tbi"),
                plot_html,
                qc_json,
            ):
                self.assertTrue(path.is_file(), path)
            self.assertFalse(Path(f"{predict_prefix}_bins.ccnv").exists())
            self.assertFalse(Path(f"{predict_prefix}_segments.ccnv").exists())

            with open(qc_json, encoding="utf-8") as f:
                qc_data = json.load(f)
            self.assertEqual(qc_data["mode"], "germline")
            self.assertEqual(qc_data["schema_version"], "2.0.0")

            with gzip.open(bins_bed, "rt") as h:
                bin_lines = h.read().splitlines()
            self.assertEqual(tuple(bin_lines[0].split("\t")), BINS_BED_HEADER)
            self.assertEqual(len(bin_lines) - 1, 468)
            self.assertEqual(bin_lines[1].split("\t")[0], "chr21")
            self.assertEqual(int(bin_lines[1].split("\t")[1]), 0)
            self.assertEqual(int(bin_lines[-1].split("\t")[2]), CHR21_LENGTH)

            with gzip.open(segments_bed, "rt") as h:
                seg_lines = h.read().splitlines()
            self.assertEqual(
                tuple(seg_lines[0].split("\t")),
                SEGMENTS_BED_HEADER,
            )

            with gzip.open(calls_bed, "rt") as h:
                call_lines = h.read().splitlines()
            self.assertEqual(
                tuple(call_lines[0].split("\t")),
                CALLS_BED_HEADER,
            )

            with pysam.TabixFile(str(bins_vcf)) as tb:
                records = list(tb.fetch("chr21", 0, CHR21_LENGTH))
                self.assertEqual(len(records), 468)

            self.assertGreater(plot_html.stat().st_size, 0)

    def test_normalize_branch_zero_null_std_mad_fallback(self) -> None:
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr2", "chr2"),
            starts=np.array([0, 100, 0, 100], dtype=np.int64),
            ends=np.array([100, 200, 100, 200], dtype=np.int64),
            counts=np.array([20.0, 30.0, 40.0, 50.0]),
            masked=np.zeros(4, dtype=np.bool_),
            pca_mean=np.full(4, 0.25),
            pca_loadings=np.zeros((1, 4)),
            nearest_indices=(
                np.array([2, 3]),
                np.array([2, 3]),
                np.array([0, 1]),
                np.array([0, 1]),
            ),
            mean_distance=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            null_mean=np.zeros(4, dtype=np.float64),
            null_std=np.zeros(4, dtype=np.float64),
        )
        config = PredictConfig(
            Path("sample"),
            Path("reference"),
            Path("output"),
            min_reference_bins=2,
        )
        result = normalize_branch(model, config)
        self.assertEqual(result.ratios.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.zscores)))

    def test_normalize_branch_zero_mad_fallback_to_std(self) -> None:
        # When >50% of bins share identical ratios, MAD degenerates to 0;
        # std fallback should produce finite, non-zero z-scores for outliers.
        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1", "chr1", "chr1", "chr2", "chr2", "chr2"),
            starts=np.array([0, 100, 200, 0, 100, 200], dtype=np.int64),
            ends=np.array([100, 200, 300, 100, 200, 300], dtype=np.int64),
            counts=np.array([20.0, 20.0, 20.0, 20.0, 20.0, 80.0]),
            masked=np.zeros(6, dtype=np.bool_),
            pca_mean=np.full(6, 0.25),
            pca_loadings=np.zeros((1, 6)),
            nearest_indices=(
                np.array([3, 4, 5]),
                np.array([3, 4, 5]),
                np.array([3, 4, 5]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
            ),
            mean_distance=np.ones(6, dtype=np.float32),
            null_mean=np.zeros(6, dtype=np.float64),
            null_std=np.zeros(6, dtype=np.float64),
        )
        config = PredictConfig(
            Path("sample"),
            Path("reference"),
            Path("output"),
            min_reference_bins=2,
        )
        result = normalize_branch(model, config)
        self.assertEqual(result.ratios.shape, (6,))
        self.assertTrue(np.all(np.isfinite(result.zscores)))
        self.assertGreater(result.zscores[5], 2.0)

    def test_resolve_sample_sex_inferences_and_validations(self) -> None:
        from ccnv.predict import _resolve_sample_sex

        sample = pa.table(
            {
                "contig": ["chr1", "chrY"],
                "start": [0, 0],
                "end": [100, 100],
                "count": [90.0, 10.0],
            }
        )
        self.assertEqual(_resolve_sample_sex("female", sample, {}, set()), "female")
        self.assertEqual(_resolve_sample_sex("male", sample, {}, set()), "male")
        self.assertIsNone(_resolve_sample_sex("autosomal", sample, {}, {"autosomal"}))

        with self.assertRaises(PredictError):
            _resolve_sample_sex(
                "autosomal", sample, {}, {"autosomal", "female_sex", "male_sex"}
            )

        meta = {
            "model": {
                "sex_signal_contig": "chrY",
                "sex_inference": {"cutoff": 0.05},
            }
        }
        self.assertEqual(
            _resolve_sample_sex(
                "autosomal", sample, meta, {"autosomal", "female_sex", "male_sex"}
            ),
            "male",
        )
        meta_high_cut = {
            "model": {
                "sex_signal_contig": "chrY",
                "sex_inference": {"cutoff": 0.15},
            }
        }
        self.assertEqual(
            _resolve_sample_sex(
                "autosomal",
                sample,
                meta_high_cut,
                {"autosomal", "female_sex", "male_sex"},
            ),
            "female",
        )


if __name__ == "__main__":
    unittest.main()
