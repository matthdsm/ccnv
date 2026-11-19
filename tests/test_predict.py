from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pysam
from helpers import (
    make_reference_table,
    make_sample_table,
    metadata_for,
    write_synthetic_sample,
)
from published_fixtures import (
    PUBLISHED_BIN_SIZE,
    ensure_published_cram,
    ensure_published_fasta,
    extract_published_samples,
)
from scipy.stats import norm

from ccnv._core import scan_alignment_bins as scan_core
from ccnv.features.fragmentomics import resolve_fragmentomics_compatibility
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    FRAGMENTOMICS_METADATA_KEY,
    PRENATAL_METADATA_KEY,
    REFERENCE_SCHEMA,
    ArtifactKind,
    BinCompatibilityError,
    BinResult,
    FragmentomicsMetadata,
    PredictConfig,
    PredictError,
    PrenatalModelMetadata,
    ReadFilters,
    ReferenceConfig,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SegmentResult,
)
from ccnv.predict import (
    PredictionInput,
    build_branch_model,
    extract_for_predict,
    load_inputs,
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
    lengths = {"chr1": 100, "chr2": 80} if include_chr2 else {"chr1": 100}
    n_bins = len(contigs)
    return make_sample_table(
        [10, 10][:n_bins],
        contigs=contigs,
        starts=[0] * n_bins,
        ends=[100, 80][:n_bins],
        contig_lengths=lengths,
        bin_size=100,
    )


def _reference_table() -> pa.Table:
    return make_reference_table(
        branches=[BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
        contigs=["chr1", "chr2"],
        starts=[0, 0],
        ends=[100, 80],
        contig_lengths={"chr1": 100, "chr2": 80},
        pca_mean=[0.5, 0.5],
        pca_loadings=[[0.0], [0.0]],
        mean_distance=[1.0, 1.0],
        null_mean=[0.0, 0.0],
        bio_dispersion=[0.02, 0.02],
        model={"branches": [BRANCH_AUTOSOMAL], "bin_size": 100},
    )


def _prenatal_reference_table() -> pa.Table:
    contig_lengths = {
        "chr1": 200_000,
        "chr2": 200_000,
        "chrX": 200_000,
        "chrY": 300_000,
    }
    prenatal_meta = PrenatalModelMetadata(
        prenatal_bin_size=100_000,
        yfrac_threshold=0.005,
        ffy_slope=1.0,
        olm_weights=[0.1],
        olm_bias=-2.2,
        ffa_logit_mode=True,
        female_chrx_ratio_mean=1.0,
        female_chrx_ratio_std=0.01,
    )
    branches_list = [
        BRANCH_AUTOSOMAL,
        BRANCH_FEMALE_SEX,
        BRANCH_PRENATAL_AUTOSOMAL,
        BRANCH_PRENATAL_FFY,
    ]
    metadata = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths=contig_lengths,
        values={
            "model": {
                "branches": branches_list,
                "bin_size": 100_000,
                "mode": "prenatal",
                "prenatal_bin_size": 100_000,
                "sex_signal_contig": "chrY",
                "sex_cutoff": 0.005,
                "sex_inference": {"cutoff": 0.005},
            },
        },
    )
    metadata[PRENATAL_METADATA_KEY] = prenatal_meta.encode()

    b_auto, b_fem = BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX
    b_pauto, b_pffy = BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_FFY
    cs, ss = ("chr1", "chr2"), (0, 100_000)
    rows = [
        *((b_auto, c, s, s + 100_000, 0.25, [0.0]) for c in cs for s in ss),
        *((b_fem, "chrX", s, s + 100_000, 0.5, [0.0]) for s in ss),
        *((b_pauto, c, s, s + 100_000, 0.0, [0.5]) for c in cs for s in ss),
        *((b_pffy, "chrY", s, s + 100_000, 0.33, [0.0]) for s in (*ss, 200_000)),
    ]
    branches, contigs, starts, ends, pca_mean, pca_loadings = (
        list(col) for col in zip(*rows, strict=True)
    )
    n_rows = len(rows)
    return make_reference_table(
        branches=branches,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        pca_mean=pca_mean,
        pca_loadings=pca_loadings,
        mean_distance=[1.0] * n_rows,
        null_mean=[0.0] * n_rows,
        bio_dispersion=[0.01] * n_rows,
        metadata=metadata,
    )


def _prenatal_sample_table() -> pa.Table:
    contig_lengths = {
        "chr1": 200_000,
        "chr2": 200_000,
        "chrX": 200_000,
        "chrY": 300_000,
    }
    grid = [
        (c, s, s + 100_000, 5 if c == "chrY" else 100)
        for c, length in contig_lengths.items()
        for s in range(0, length, 100_000)
    ]
    contigs, starts, ends, counts = (list(col) for col in zip(*grid, strict=True))
    tbl = make_sample_table(
        counts,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        bin_size=100_000,
    )
    meta = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": 100_000,
            "contig_order": list(contig_lengths.keys()),
            "qc": {
                "effective_reads_examined": 500,
                "effective_reads_kept": 400,
                "autosomal_reads_kept": 380,
                "sex_reads_kept": 20,
            },
            "filters": asdict(ReadFilters()),
        },
    )
    return tbl.replace_schema_metadata(meta)


class TestCalling(unittest.TestCase):
    def test_calls_symmetric_z_and_ratio_thresholds(self) -> None:
        def _single_seg_call(
            ratio: float, z: float, *, ratio_threshold: float | None = None
        ) -> str | None:
            bins = (
                BinResult(
                    BRANCH_AUTOSOMAL,
                    ("chr1",) * 8,
                    np.arange(8, dtype=np.int64) * 100,
                    np.arange(1, 9, dtype=np.int64) * 100,
                    np.full(8, ratio, dtype=np.float64),
                    np.full(8, z, dtype=np.float64),
                    np.full(8, 25.0, dtype=np.float64),
                    np.full(8, 10, dtype=np.int64),
                    np.zeros(8, dtype=np.bool_),
                ),
            )
            segs = segment_and_call(
                bins,
                {"chr1": 800},
                penalty=1.0,
                z_threshold=3.0,
                ratio_threshold=ratio_threshold,
            )
            return segs[0].call

        self.assertEqual(_single_seg_call(0.585, 3.0), "gain")
        self.assertEqual(_single_seg_call(-0.95, -3.0), "loss")
        self.assertIsNone(_single_seg_call(0.18, 4.0, ratio_threshold=0.35))
        self.assertIsNone(_single_seg_call(0.05, 0.5))

    def test_segment_and_call_stouffer_zscore_accumulates_evidence_and_corrects_autocorr(
        self,
    ) -> None:
        z = np.full(25, 1.8, dtype=np.float64) + np.tile([0.05, -0.05], 13)[:25]
        bins = (
            BinResult(
                BRANCH_AUTOSOMAL,
                ("chr1",) * 25,
                np.arange(25, dtype=np.int64) * 100,
                np.arange(1, 26, dtype=np.int64) * 100,
                np.full(25, 0.585, dtype=np.float64),
                z,
                np.ones(25, dtype=np.float64),
                np.full(25, 10, dtype=np.int64),
                np.zeros(25, dtype=np.bool_),
            ),
        )
        segs = segment_and_call(
            bins, {"chr1": 2500}, penalty=1.0, z_threshold=3.0, ratio_threshold=None
        )
        self.assertEqual(len(segs), 1)
        self.assertGreater(segs[0].zscore, 8.0)
        self.assertEqual(segs[0].call, "gain")

    def test_segment_and_call_preserves_zero_usable_bins_segment(self) -> None:
        starts = np.arange(8, dtype=np.int64) * 100
        ends = np.arange(1, 9, dtype=np.int64) * 100
        ratios = np.array([0.585] * 4 + [0.0] * 4, dtype=np.float64)
        bins_neutral_tail = (
            BinResult(
                BRANCH_AUTOSOMAL,
                ("chr1",) * 8,
                starts,
                ends,
                ratios,
                np.array([3.5] * 4 + [0.0] * 4, dtype=np.float64),
                np.full(8, 200.0, dtype=np.float64),
                np.full(8, 10, dtype=np.int64),
                np.zeros(8, dtype=np.bool_),
            ),
        )
        segs = segment_and_call(
            bins_neutral_tail, {"chr1": 800}, penalty=0.5, z_threshold=3.0
        )
        self.assertEqual(len(segs), 2)
        self.assertEqual((segs[1].start, segs[1].end), (400, 800))
        self.assertEqual(segs[1].zscore, 0.0)
        self.assertIsNone(segs[1].call)

        bins_nan_tail = (
            BinResult(
                BRANCH_AUTOSOMAL,
                ("chr1",) * 8,
                starts,
                ends,
                ratios,
                np.array([3.5] * 4 + [np.nan] * 4, dtype=np.float64),
                np.full(8, 200.0, dtype=np.float64),
                np.full(8, 10, dtype=np.int64),
                np.zeros(8, dtype=np.bool_),
            ),
        )
        trimmed = segment_and_call(
            bins_nan_tail, {"chr1": 800}, penalty=0.5, z_threshold=3.0
        )
        self.assertEqual(len(trimmed), 1)
        self.assertEqual((trimmed[0].start, trimmed[0].end), (0, 400))


class TestCoordinatesAndOutputs(unittest.TestCase):
    def test_bed_is_zero_based_half_open(self) -> None:
        bins = (
            BinResult(
                BRANCH_AUTOSOMAL,
                ("chr1", "chr1"),
                np.array([0, 95]),
                np.array([95, 100]),
                np.array([0.5, np.nan]),
                np.array([4.0, np.nan]),
                np.array([1.0, 0.0]),
                np.array([10, 0]),
                np.array([False, True]),
            ),
        )
        segments = (
            SegmentResult("chr1", 0, 95, 0.5, 4.0, 1.0, 1, "gain"),
            SegmentResult("chr1", 95, 100, 0.0, 0.0, 0.0, 1, None),
        )
        inputs = PredictionInput(
            SampleBuffers.from_table(_sample_table()),
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
            bins_rows[2].split("\t")[:6], ["chr1", "95", "100", "chr1:95-100", ".", "."]
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
                BRANCH_AUTOSOMAL,
                ("chr1", "chr1", "chr1"),
                np.array([0, 100, 200]),
                np.array([100, 200, 250]),
                np.zeros(3),
                np.zeros(3),
                np.ones(3),
                np.full(3, 10),
                np.zeros(3, dtype=np.bool_),
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
            write_sample_parquet(_sample_table(include_chr2=False), sample_path)
            write_reference_parquet(_reference_table(), reference_path)

            with self.assertRaisesRegex(PredictError, "missing.*chr2"):
                load_inputs(sample_path, reference_path)

    def test_load_inputs_rebins_sample_without_modifying_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"

            sample_50 = make_sample_table(
                [5, 5, 6, 4],
                contigs=["chr1", "chr1", "chr2", "chr2"],
                starts=[0, 50, 0, 50],
                ends=[50, 100, 50, 80],
                contig_lengths={"chr1": 100, "chr2": 80},
                bin_size=50,
            )
            write_sample_parquet(sample_50, sample_path)
            write_reference_parquet(_reference_table(), reference_path)

            ref_bytes_before = reference_path.read_bytes()
            ref_mtime_before = reference_path.stat().st_mtime_ns
            inputs = load_inputs(sample_path, reference_path)

            self.assertEqual(inputs.sample.num_rows, 2)
            self.assertEqual(inputs.sample.meta.bin_size, 100)
            np.testing.assert_allclose(inputs.sample.count, [10.0, 10.0])
            self.assertEqual(reference_path.read_bytes(), ref_bytes_before)
            self.assertEqual(reference_path.stat().st_mtime_ns, ref_mtime_before)

    def test_load_inputs_rejects_contig_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"

            sample_mismatch = make_sample_table(
                [10, 5, 10],
                contigs=["chr1", "chr1", "chr2"],
                starts=[0, 100, 0],
                ends=[100, 150, 80],
                contig_lengths={"chr1": 150, "chr2": 80},
                bin_size=100,
            )
            write_sample_parquet(sample_mismatch, sample_path)
            write_reference_parquet(_reference_table(), reference_path)

            with self.assertRaisesRegex(PredictError, "contig length mismatch"):
                load_inputs(sample_path, reference_path)

    def test_build_branch_model_rejects_mismatched_sample_coordinates(self) -> None:
        ref = _reference_table()
        bad_sample = SampleBuffers.from_table(_sample_table(include_chr2=False))
        inputs = PredictionInput(
            bad_sample, ref, {"chr1": 100, "chr2": 80}, None, (BRANCH_AUTOSOMAL,)
        )
        with self.assertRaises(PredictError):
            build_branch_model(inputs, BRANCH_AUTOSOMAL)

    def test_sample_buffers_rebin_and_branch_model_without_arrow_table(self) -> None:
        z32 = np.zeros(4, dtype=np.uint32)
        bufs_50 = SampleBuffers(
            meta=SampleMetadata(
                bin_size=50,
                contig_order=("chr1", "chr2"),
                contig_lengths={"chr1": 100, "chr2": 80},
                qc={},
                filters={},
            ),
            count=np.array([5, 5, 6, 4], dtype=np.uint32),
            paired_count=z32,
            frag_len_sum=np.zeros(4, dtype=np.uint64),
            short_frag_count=z32,
            endpoint_count=z32,
            frag_gc_sum=np.zeros(4, dtype=np.float32),
            frag_bands=None,
        )
        bufs_100 = rebin_sample_buffers(bufs_50, 100)
        self.assertEqual(bufs_100.meta.bin_size, 100)
        self.assertIsNone(bufs_100.frag_bands)
        np.testing.assert_array_equal(
            bufs_100.count, np.array([10, 10], dtype=np.uint32)
        )

        bm = build_branch_model(
            PredictionInput(
                bufs_100,
                _reference_table(),
                {"chr1": 100, "chr2": 80},
                None,
                (BRANCH_AUTOSOMAL,),
            ),
            BRANCH_AUTOSOMAL,
        )
        np.testing.assert_allclose(bm.counts, [10.0, 10.0])


class TestModeGuardrails(unittest.TestCase):
    def test_prenatal_mode_with_germline_ref_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            write_sample_parquet(_sample_table(), sample_path)
            write_reference_parquet(_reference_table(), reference_path)

            expected_msg = (
                "reference was built in germline mode and lacks prenatal models; "
                "rebuild reference with --mode prenatal or run predict with --mode germline"
            )
            with self.assertRaisesRegex(PredictError, expected_msg):
                load_inputs(sample_path, reference_path, mode="prenatal")

    def test_germline_mode_with_prenatal_ref_warns_and_filters_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            write_sample_parquet(_prenatal_sample_table(), sample_path)
            write_reference_parquet(_prenatal_reference_table(), reference_path)

            with self.assertLogs("ccnv.predict", level="WARNING") as cm:
                inputs = load_inputs(sample_path, reference_path, mode="germline")

            self.assertTrue(
                any(
                    "prenatal models are ignored in germline mode" in log
                    for log in cm.output
                )
            )
            self.assertIn(BRANCH_AUTOSOMAL, inputs.branches)
            self.assertNotIn(BRANCH_PRENATAL_AUTOSOMAL, inputs.branches)
            self.assertNotIn(BRANCH_PRENATAL_FFY, inputs.branches)


class TestPrenatalPredictPipeline(unittest.TestCase):
    def test_predict_prenatal_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sample_path = directory / "sample.ccnv"
            reference_path = directory / "reference.ccnv"
            write_sample_parquet(_prenatal_sample_table(), sample_path)
            write_reference_parquet(_prenatal_reference_table(), reference_path)

            outputs = run_predict(
                PredictConfig(
                    sample=sample_path,
                    reference=reference_path,
                    prefix=directory / "prenatal_out",
                    mode="prenatal",
                    min_reference_bins=1,
                )
            )
            self.assertIsNotNone(outputs.prenatal_result)
            self.assertIsNotNone(outputs.qc_json_path)
            if outputs.qc_json_path is None:
                self.fail("expected qc_json_path")
            self.assertTrue(outputs.qc_json_path.is_file())

            qc_data = json.loads(outputs.qc_json_path.read_text(encoding="utf-8"))
            self.assertEqual(qc_data["mode"], "prenatal")
            self.assertEqual(qc_data["sex"], "male")
            self.assertIsNotNone(qc_data["prenatal"])
            self.assertEqual(qc_data["prenatal"]["fetal_sex"], "M")
            self.assertGreater(qc_data["prenatal"]["active_ff"], 0.0)
            self.assertIn("read_metrics", qc_data)
            self.assertIn("cnv_summary", qc_data)

            chry_segs = [s for s in outputs.segments if s.contig == "chrY"]
            self.assertGreaterEqual(len(chry_segs), 1)
            for seg in chry_segs:
                self.assertIsNone(seg.expected_ratio)
                self.assertIsNone(seg.implied_ff)
            for seg in (s for s in outputs.segments if s.contig != "chrY"):
                self.assertIsNotNone(seg.expected_ratio)
                self.assertIsNotNone(seg.implied_ff)
                self.assertIsInstance(seg.discordant, bool)

    def test_predict_germline_qc_records_inferred_and_null_sex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            auto_sample = root / "auto_sample.ccnv"
            auto_ref = root / "auto_ref.ccnv"
            write_sample_parquet(_sample_table(), auto_sample)
            write_reference_parquet(_reference_table(), auto_ref)
            auto_out = run_predict(
                PredictConfig(
                    sample=auto_sample,
                    reference=auto_ref,
                    prefix=root / "auto_pred",
                    min_reference_bins=1,
                    plot=False,
                    vcf=False,
                )
            )
            if auto_out.qc_json_path is None:
                self.fail("expected qc_json_path")
            auto_qc = json.loads(auto_out.qc_json_path.read_text(encoding="utf-8"))
            self.assertIsNone(auto_qc["sex"])

            lengths = {"chr1": 200, "chr2": 200, "chrX": 100, "chrY": 100}
            rows = [
                (BRANCH_AUTOSOMAL, "chr1", 0, 100),
                (BRANCH_AUTOSOMAL, "chr1", 100, 200),
                (BRANCH_AUTOSOMAL, "chr2", 0, 100),
                (BRANCH_AUTOSOMAL, "chr2", 100, 200),
                (BRANCH_FEMALE_SEX, "chrX", 0, 100),
                (BRANCH_MALE_SEX, "chrX", 0, 100),
                (BRANCH_MALE_SEX, "chrY", 0, 100),
            ]
            branches, contigs, starts, ends = (
                list(col) for col in zip(*rows, strict=True)
            )
            n_r = len(rows)
            sex_ref_tbl = make_reference_table(
                branches=branches,
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths=lengths,
                pca_mean=[0.2] * n_r,
                pca_loadings=[[0.0]] * n_r,
                mean_distance=[1.0] * n_r,
                null_mean=[0.0] * n_r,
                bio_dispersion=[0.02] * n_r,
                model={
                    "branches": [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX],
                    "bin_size": 100,
                    "sex_signal_contig": "chrY",
                    "sex_cutoff": 0.05,
                },
            )
            sex_ref = root / "sex_ref.ccnv"
            write_reference_parquet(sex_ref_tbl, sex_ref)

            for expected_sex, y_cnt in (("female", 0), ("male", 40)):
                s_tbl = make_sample_table(
                    [50, 50, 50, 50, 50 if expected_sex == "female" else 25, y_cnt],
                    contigs=["chr1", "chr1", "chr2", "chr2", "chrX", "chrY"],
                    starts=[0, 100, 0, 100, 0, 0],
                    ends=[100, 200, 100, 200, 100, 100],
                    contig_lengths=lengths,
                    bin_size=100,
                )
                s_path = root / f"{expected_sex}.ccnv"
                write_sample_parquet(s_tbl, s_path)
                out = run_predict(
                    PredictConfig(
                        sample=s_path,
                        reference=sex_ref,
                        prefix=root / f"pred_{expected_sex}",
                        min_reference_bins=1,
                        plot=False,
                        vcf=False,
                    )
                )
                if out.qc_json_path is None:
                    self.fail("expected qc_json_path")
                qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                self.assertEqual(qc["sex"], expected_sex)


class TestPredictAlignmentInputs(unittest.TestCase):
    def test_load_inputs_polymorphic_cram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1, s2 = extract_published_samples(root)
            ref_path = root / "cohort_ref.ccnv"
            build_reference(
                ReferenceConfig(
                    entries=tuple(SamplesheetEntry(filepath=p) for p in (s1, s2)),
                    prefix=root / "cohort_ref",
                    bin_size=PUBLISHED_BIN_SIZE,
                    min_cohort_size=2,
                )
            )
            cram = ensure_published_cram()
            fasta = ensure_published_fasta()
            with patch.object(
                SampleBuffers,
                "to_table",
                side_effect=AssertionError("must not materialize SAMPLE_SCHEMA table"),
            ):
                inputs = load_inputs(
                    cram,
                    ref_path,
                    sex="autosomal",
                    mode="germline",
                    fasta=fasta,
                )
            self.assertIsInstance(inputs.sample, SampleBuffers)
            self.assertIsNone(inputs.sample.frag_bands)
            self.assertIsNotNone(inputs.sample.meta.fragmentomics)
            self.assertGreater(inputs.sample.num_rows, 0)

    def test_fragmentomics_compatibility_and_observed_qc_counts(self) -> None:
        ref_meta_obj = ReferenceMetadata(
            bin_size=10000,
            contig_lengths={"chr1": 20000},
            branches=(BRANCH_AUTOSOMAL,),
            fragmentomics_corrected=True,
        )
        restored_meta = ReferenceMetadata.from_schema(
            REFERENCE_SCHEMA.with_metadata(ref_meta_obj.build_arrow_metadata())
        )
        self.assertTrue(restored_meta.fragmentomics_corrected)

        se_sample = make_sample_table(
            [100, 120],
            contigs=["chr1", "chr1"],
            starts=[0, 10000],
            ends=[10000, 20000],
            contig_lengths={"chr1": 20000},
            bin_size=10000,
            endpoint_counts=[50, 60],
        )
        pe_frag_meta = FragmentomicsMetadata(
            True,
            220,
            166.0,
            143.0,
            0.0,
            0.35,
            0.05,
            0.01,
            0.04,
            1.0,
            0.90,
            [1] * 571,
            [0] * 571,
            [0] * 571,
            [1] * 256,
        )
        pe_sample = make_sample_table(
            [100, 120],
            contigs=["chr1", "chr1"],
            starts=[0, 10000],
            ends=[10000, 20000],
            contig_lengths={"chr1": 20000},
            bin_size=10000,
            fragment_lengths=[[135] * 100, [135] * 120],
            endpoint_counts=[50, 60],
        )
        pe_meta_dict = dict(pe_sample.schema.metadata or {})
        pe_meta_dict[FRAGMENTOMICS_METADATA_KEY] = json.dumps(
            asdict(pe_frag_meta)
        ).encode()
        pe_sample = pe_sample.replace_schema_metadata(pe_meta_dict)

        def _make_ref(model_dict: dict[str, object]) -> pa.Table:
            return make_reference_table(
                branches=[BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
                contigs=["chr1", "chr1"],
                starts=[0, 10000],
                ends=[10000, 20000],
                contig_lengths={"chr1": 20000},
                pca_mean=[100.0, 120.0],
                mean_distance=[0.5, 0.5],
                null_mean=[0.0, 0.0],
                bio_dispersion=[0.0025, 0.0025],
                model=model_dict,
                sample_count=1,
            )

        legacy_ref = _make_ref({"bin_size": 10000})
        se_ref = _make_ref({"bin_size": 10000, "fragmentomics_corrected": False})
        pe_ref = _make_ref({"bin_size": 10000, "fragmentomics_corrected": True})
        flat_ref = _make_ref({"bin_size": 10000, "flat": True})

        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, legacy_ref))
        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, se_ref))
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(pe_sample, legacy_ref)
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(pe_sample, se_ref)

        self.assertTrue(resolve_fragmentomics_compatibility(pe_sample, pe_ref))
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(se_sample, pe_ref)

        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, flat_ref))
        self.assertTrue(resolve_fragmentomics_compatibility(pe_sample, flat_ref))

        se_in = PredictionInput(
            SampleBuffers.from_table(se_sample),
            se_ref,
            {"chr1": 20000},
            None,
            (BRANCH_AUTOSOMAL,),
        )
        se_bm = build_branch_model(se_in, BRANCH_AUTOSOMAL)
        self.assertIsNone(se_bm.ref_frag_len)
        self.assertIsNone(se_bm.ref_ep_rate)

        grid = np.arange(30, 601, dtype=np.float64)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_fet /= np.sum(p_fet)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_mat /= np.sum(p_mat)
        prenatal_meta = PrenatalModelMetadata(
            20000,
            0.005,
            1.0,
            [0.0],
            -2.0,
            1.0,
            0.01,
            p_fetal_density=p_fet.tolist(),
            p_maternal_density=p_mat.tolist(),
        )
        p_branches = [BRANCH_AUTOSOMAL, BRANCH_PRENATAL_AUTOSOMAL, BRANCH_PRENATAL_FFY]
        p_ranks = {
            BRANCH_AUTOSOMAL: 0,
            BRANCH_PRENATAL_AUTOSOMAL: 1,
            BRANCH_PRENATAL_FFY: 1,
        }
        prenatal_rows = make_reference_table(
            branches=[BRANCH_AUTOSOMAL, *p_branches, BRANCH_PRENATAL_FFY],
            contigs=["chr1", "chr1", "chr1", "chrY", "chrY"],
            starts=[0, 10000, 0, 0, 10000],
            ends=[10000, 20000, 20000, 10000, 20000],
            contig_lengths={"chr1": 20000, "chrY": 20000},
            pca_mean=[100.0, 120.0, 220.0, 1.0, 1.0],
            pca_loadings=[[], [], [1.0], [0.5], [0.5]],
            mean_distance=[0.5, 0.5, 0.0, 0.0, 0.0],
            null_mean=[0.0] * 5,
            bio_dispersion=[0.0025, 0.0025, 0.0, 0.0, 0.0],
            model={
                "branches": p_branches,
                "bin_size": 10000,
                "mode": "prenatal",
                "prenatal_bin_size": 20000,
                "fragmentomics_corrected": True,
                "branch_pca_ranks": p_ranks,
            },
            sample_count=1,
            extra_metadata={"prenatal": prenatal_meta.to_dict()},
        )
        pe_prenatal_sample = make_sample_table(
            counts=[100, 120, 0, 0],
            contigs=["chr1", "chr1", "chrY", "chrY"],
            starts=[0, 10000, 0, 10000],
            ends=[10000, 20000, 10000, 20000],
            contig_lengths={"chr1": 20000, "chrY": 20000},
            bin_size=10000,
            fragment_lengths=[[135] * 100, [135] * 120, [], []],
            endpoint_counts=[50, 60, 0, 0],
        )
        pe_prenatal_meta = dict(pe_prenatal_sample.schema.metadata or {})
        pe_prenatal_meta[FRAGMENTOMICS_METADATA_KEY] = json.dumps(
            asdict(pe_frag_meta)
        ).encode()
        pe_prenatal_sample = pe_prenatal_sample.replace_schema_metadata(
            pe_prenatal_meta
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sm_path = root / "pe_sample.ccnv"
            rf_path = root / "pe_ref.ccnv"
            write_sample_parquet(pe_prenatal_sample, sm_path)
            write_reference_parquet(prenatal_rows, rf_path)
            outputs = run_predict(
                PredictConfig(
                    sample=sm_path,
                    reference=rf_path,
                    prefix=root / "out",
                    mode="prenatal",
                    plot=False,
                )
            )
            self.assertIsNotNone(outputs.qc_json_path)
            if outputs.qc_json_path is None:
                self.fail("expected qc_json_path")
            qc_payload = json.loads(outputs.qc_json_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                qc_payload["read_metrics"]["mean_bin_count"], 55.0, places=4
            )
            self.assertAlmostEqual(
                qc_payload["read_metrics"]["median_bin_count"], 50.0, places=4
            )

    def test_extract_for_predict_contig_naming_resolution_and_default_mapq_1(
        self,
    ) -> None:
        ref_partial_filters = ReferenceMetadata(
            bin_size=200,
            contig_lengths={"1": 1200, "2": 1200, "X": 800, "Y": 800},
            branches=(BRANCH_AUTOSOMAL,),
            filters={"exclude_duplicates": True},
        )
        resolved_filters = ReadFilters.from_dict(
            ref_partial_filters.filters or {}, strict=False
        )
        self.assertEqual(resolved_filters.min_mapping_quality, 1)
        self.assertEqual(ReadFilters().min_mapping_quality, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ucsc_bam = tmp_path / "ucsc_mapq5.bam"
            header = {
                "HD": {"VN": "1.6", "SO": "coordinate"},
                "SQ": [
                    {"SN": "chr1", "LN": 1200},
                    {"SN": "chr2", "LN": 1200},
                    {"SN": "chrX", "LN": 800},
                    {"SN": "chrY", "LN": 800},
                ],
            }
            with pysam.AlignmentFile(str(ucsc_bam), "wb", header=header) as outf:
                for tid in (0, 1, 2):
                    a = pysam.AlignedSegment(outf.header)
                    a.query_name = f"mq5_{tid}"
                    a.query_sequence = "ACGT" * 12 + "AC"
                    a.flag = 99
                    a.reference_id = tid
                    a.reference_start = 50
                    a.mapping_quality = 5
                    a.cigar = ((0, 50),)
                    a.next_reference_id = tid
                    a.next_reference_start = 160
                    a.template_length = 160
                    a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                    outf.write(a)
            pysam.index(str(ucsc_bam))

            core_default = scan_core(str(ucsc_bam), [("1", 1200)], 200)
            self.assertEqual(int(np.sum(np.asarray(core_default["count"]))), 1)

            extracted = extract_for_predict(ucsc_bam, ref_partial_filters, threads=1)
            self.assertEqual(list(extracted.meta.contig_order), ["1", "2", "X", "Y"])
            self.assertEqual(int(np.sum(extracted.count)), 3)

            cohort_paths: list[Path] = []
            for idx in range(4):
                p = tmp_path / f"ens_{idx}.ccnv"
                write_synthetic_sample(
                    p,
                    [35 + idx] * 12,
                    contigs=["1"] * 6 + ["2"] * 6,
                    starts=[0, 200, 400, 600, 800, 1000] * 2,
                    ends=[200, 400, 600, 800, 1000, 1200] * 2,
                    contig_lengths={"1": 1200, "2": 1200},
                    bin_size=200,
                    fragment_lengths=[[160] * (35 + idx)] * 12,
                    endpoint_counts=[4] * 12,
                )
                cohort_paths.append(p)
            ens_ref_path = build_reference(
                ReferenceConfig(
                    entries=tuple(SamplesheetEntry(filepath=p) for p in cohort_paths),
                    prefix=tmp_path / "ens_ref",
                    bin_size=200,
                    min_cohort_size=4,
                    sex_contigs=(),
                )
            )
            inputs = load_inputs(ucsc_bam, ens_ref_path, mode="germline", threads=2)
            self.assertEqual(inputs.contig_lengths, {"1": 1200, "2": 1200})

            bad_len_meta = ReferenceMetadata(
                bin_size=200, contig_lengths={"1": 1400}, branches=(BRANCH_AUTOSOMAL,)
            )
            with self.assertRaisesRegex(
                BinCompatibilityError, "contig length mismatch"
            ):
                extract_for_predict(ucsc_bam, bad_len_meta)
            with self.assertRaisesRegex(PredictError, "contig length mismatch"):
                extract_for_predict(ucsc_bam, bad_len_meta)


if __name__ == "__main__":
    unittest.main()
