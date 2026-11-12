"""Unit tests for ccnv.qc subpackage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_reference_table, make_sample_table

from ccnv.models import (
    BRANCH_AUTOSOMAL,
    FetalFractionResult,
    SCAResult,
    SegmentResult,
)
from ccnv.qc.predict import (
    build_predict_qc_payload,
    write_predict_qc_json,
)
from ccnv.qc.reference import (
    compute_signal_contig_metrics,
    evaluate_autosomal_or_female_verdict,
    evaluate_male_verdict,
    run_reference_qc_on_table,
    write_reference_qc_json,
)
from ccnv.qc.sample import (
    QC_COUNTERS,
    build_sample_qc,
    empty_qc,
)


class TestQCSample(unittest.TestCase):
    def test_empty_qc(self) -> None:
        qc = empty_qc()
        self.assertEqual(tuple(qc.keys()), QC_COUNTERS)
        self.assertTrue(all(v == 0 for v in qc.values()))

    def test_build_sample_qc_defaults(self) -> None:
        qc = build_sample_qc(reads_seen=100, reads_kept=95)
        self.assertEqual(qc["reads_seen"], 100)
        self.assertEqual(qc["reads_kept"], 95)
        self.assertEqual(qc["effective_reads_examined"], 100)
        self.assertEqual(qc["effective_reads_kept"], 95)

    def test_build_sample_qc_explicit(self) -> None:
        qc = build_sample_qc(
            reads_seen=100,
            reads_kept=95,
            effective_reads_examined=80,
            effective_reads_kept=75,
        )
        self.assertEqual(qc["reads_seen"], 100)
        self.assertEqual(qc["effective_reads_examined"], 80)
        self.assertEqual(qc["effective_reads_kept"], 75)


class TestQCReference(unittest.TestCase):
    def test_compute_signal_contig_metrics(self) -> None:
        contigs = np.array(["chr1", "chrY", "chrY"])
        mean_d = np.array([0.5, 1.2, 1.4], dtype=np.float64)
        n_refs = np.array([200, 200, 200], dtype=np.int64)

        metrics = compute_signal_contig_metrics(
            contigs, "chrY", mean_d, n_refs, cutoff_outlier=2.0, min_ref_bins=100
        )
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["signal_contig"], "chrY")
        self.assertEqual(metrics["n_bins"], 2)
        self.assertEqual(metrics["n_valid"], 2)
        self.assertAlmostEqual(float(metrics["mean_of_means"]), 1.3)

    def test_evaluate_autosomal_verdict(self) -> None:
        good = {
            "n_valid": 100,
            "n_low_refs": 0,
            "outlier_pct": 0.5,
            "std_of_means": 0.8,
        }
        verdict, _ = evaluate_autosomal_or_female_verdict(good, min_ref_bins=150)
        self.assertEqual(verdict, "PASS")

        warn = {
            "n_valid": 100,
            "n_low_refs": 5,
            "outlier_pct": 2.0,
            "std_of_means": 2.5,
        }
        verdict, _msg = evaluate_autosomal_or_female_verdict(warn, min_ref_bins=150)
        self.assertEqual(verdict, "WARN")

        fail = {
            "n_valid": 100,
            "n_low_refs": 0,
            "outlier_pct": 0.0,
            "std_of_means": 15.0,
        }
        verdict, _msg = evaluate_autosomal_or_female_verdict(fail, min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")

    def test_evaluate_male_verdict(self) -> None:
        good = {
            "n_valid": 100,
            "n_low_refs": 0,
            "mean_of_means": 1.0,
            "outlier_pct": 0.5,
            "signal_contig_metrics": {
                "n_valid": 10,
                "usable_pct": 90.0,
                "mean_of_means": 2.0,
            },
        }
        verdict, _ = evaluate_male_verdict(good, min_ref_bins=150)
        self.assertEqual(verdict, "PASS")

        poor_chry = {
            "n_valid": 100,
            "n_low_refs": 0,
            "mean_of_means": 1.0,
            "outlier_pct": 0.5,
            "signal_contig_metrics": {
                "n_valid": 10,
                "usable_pct": 10.0,
                "mean_of_means": 2.0,
            },
        }
        verdict, _msg = evaluate_male_verdict(poor_chry, min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")

    def test_write_reference_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sub" / "ref_qc.json"
            write_reference_qc_json(json_path, {"verdict": "PASS", "score": 0.99})
            self.assertTrue(json_path.is_file())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["verdict"], "PASS")

    def test_run_reference_qc_on_table(self) -> None:
        n_bins = 200
        table = make_reference_table(
            branches=[BRANCH_AUTOSOMAL] * n_bins,
            contigs=["chr1"] * n_bins,
            starts=np.arange(n_bins) * 100,
            ends=(np.arange(n_bins) + 1) * 100,
            contig_lengths={"chr1": 20000},
            pca_mean=np.zeros(n_bins, dtype=np.float32),
            mean_distance=np.ones(n_bins, dtype=np.float32),
            null_mean=np.zeros(n_bins, dtype=np.float32),
            bio_dispersion=np.ones(n_bins, dtype=np.float32),
            sample_identities=[{"path": "s1"}],
            model={
                "mode": "germline",
                "bin_size": 100,
                "nearest_bins": 300,
                "flat": False,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "test_ref"
            outputs = run_reference_qc_on_table(
                table, prefix=prefix, write_json=True, emit_stderr=False
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertTrue(outputs.json_path.is_file())


class TestQCPredict(unittest.TestCase):
    def test_build_predict_qc_payload_germline(self) -> None:
        table = make_sample_table([10, 20, 30], bin_size=100)
        segments = [
            SegmentResult("chr1", 0, 100, 0.5, 3.2, 1.0, 1, "gain"),
            SegmentResult("chr1", 100, 300, -0.5, -3.2, 2.0, 2, "loss"),
        ]
        payload = build_predict_qc_payload("germline", table, segments)
        self.assertEqual(payload["mode"], "germline")
        cnv_summary = payload["cnv_summary"]
        if not isinstance(cnv_summary, dict):
            self.fail("expected cnv_summary dict")
        self.assertEqual(cnv_summary["total_segments"], 2)
        self.assertEqual(cnv_summary["gain_calls"], 1)
        self.assertEqual(cnv_summary["loss_calls"], 1)
        self.assertIsNone(payload["prenatal"])

    def test_build_predict_qc_payload_prenatal(self) -> None:
        table = make_sample_table([10, 20], bin_size=100)
        sca = SCAResult("XY", False, 0.95, 0.1, 0.0)
        from ccnv.models import PrenatalLoDMetrics

        lod = PrenatalLoDMetrics(lod_ff_35mb=0.015, lod_ff_3mb=0.048)
        ff_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.12,
            active_ff_source="FFY",
            ffy=0.12,
            ffx=0.11,
            ffa=0.10,
            chrx_ratio=0.94,
            chry_score=0.015,
            yfrac_threshold=0.002,
            sca=sca,
            low_ff_warning=False,
            warnings=(),
            active_ff_se=0.008,
            lod_metrics=lod,
            effective_ff=0.14,
        )
        segments = [
            SegmentResult(
                "chr1",
                0,
                100,
                0.5,
                3.2,
                1.0,
                1,
                "gain",
                origin_call="FETAL_CONSTITUTIONAL",
            ),
            SegmentResult(
                "chr1",
                100,
                300,
                -0.5,
                -3.2,
                2.0,
                2,
                "loss",
                origin_call="MATERNAL_UNINHERITED",
            ),
        ]
        payload = build_predict_qc_payload(
            "prenatal", table, segments, prenatal_result=ff_res
        )
        self.assertEqual(payload["mode"], "prenatal")
        prenatal_payload = payload["prenatal"]
        if not isinstance(prenatal_payload, dict):
            self.fail("expected prenatal dict")
        self.assertEqual(prenatal_payload["fetal_sex"], "M")
        self.assertEqual(prenatal_payload["active_ff"], 0.12)
        self.assertEqual(prenatal_payload["effective_ff"], 0.14)
        self.assertEqual(prenatal_payload["active_ff_se"], 0.008)
        self.assertEqual(prenatal_payload["lod_ff_35mb"], 0.015)
        self.assertEqual(prenatal_payload["lod_ff_3mb"], 0.048)
        self.assertEqual(
            prenatal_payload["origin_summary"],
            {"FETAL_CONSTITUTIONAL": 1, "MATERNAL_UNINHERITED": 1},
        )

    def test_write_predict_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sub" / "predict_qc.json"
            payload = {
                "mode": "germline",
                "float_val": np.float64(0.123456789),
                "int_val": np.int64(42),
                "bool_val": np.bool_(True),
            }
            write_predict_qc_json(json_path, payload)
            self.assertTrue(json_path.is_file())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["float_val"], 0.123457)
            self.assertEqual(loaded["int_val"], 42)
            self.assertTrue(loaded["bool_val"])


if __name__ == "__main__":
    unittest.main()
