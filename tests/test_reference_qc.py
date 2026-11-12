import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_reference_table

from ccnv.models import ReferenceQCConfig
from ccnv.qc.reference import (
    CHRY_MEAN_DIST_FAIL,
    CHRY_MIN_USABLE_PCT_FAIL,
    CHRY_MIN_USABLE_PCT_WARN,
    ReferenceQCError,
    compute_signal_contig_metrics,
    evaluate_autosomal_or_female_verdict,
    evaluate_male_verdict,
    run_reference_qc_on_table,
)
from ccnv.ref_qc import run_reference_qc


class TestReferenceQC(unittest.TestCase):
    def test_compute_signal_contig_metrics_missing(self) -> None:
        contigs = ["chr1", "chr2"]
        mean_d = np.array([1.0, 2.0], dtype=np.float64)
        n_refs = np.array([200, 200], dtype=np.int64)
        res = compute_signal_contig_metrics(contigs, "chrY", mean_d, n_refs, 5.0, 150)
        self.assertIsNone(res)

    def test_compute_signal_contig_metrics_present(self) -> None:
        contigs = ["chr1", "chrY", "chrY"]
        mean_d = np.array([1.0, 2.0, 6.0], dtype=np.float64)
        n_refs = np.array([200, 100, 160], dtype=np.int64)
        res = compute_signal_contig_metrics(contigs, "chrY", mean_d, n_refs, 5.0, 150)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertEqual(res["signal_contig"], "chrY")
        self.assertEqual(res["n_bins"], 2)
        self.assertEqual(res["n_valid"], 2)
        self.assertAlmostEqual(float(res["mean_of_means"]), 4.0)
        self.assertEqual(res["n_mean_outlier"], 1)
        self.assertEqual(res["n_low_refs"], 1)
        self.assertEqual(res["n_usable"], 1)
        self.assertAlmostEqual(float(res["usable_pct"]), 50.0)

    def test_evaluate_autosomal_verdict_pass(self) -> None:
        m = {
            "n_valid": 1000,
            "std_of_means": 1.2,
            "outlier_pct": 0.5,
            "n_low_refs": 0,
        }
        verdict, msg = evaluate_autosomal_or_female_verdict(m, min_ref_bins=150)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(msg, "")

    def test_evaluate_autosomal_verdict_warn_and_fail(self) -> None:
        m_warn = {
            "n_valid": 1000,
            "std_of_means": 2.5,
            "outlier_pct": 0.5,
            "n_low_refs": 0,
        }
        verdict, msg = evaluate_autosomal_or_female_verdict(m_warn, min_ref_bins=150)
        self.assertEqual(verdict, "WARN")
        self.assertIn("std(per-bin mean dist)", msg)

        m_fail = {
            "n_valid": 1000,
            "std_of_means": 11.0,
            "outlier_pct": 0.5,
            "n_low_refs": 0,
        }
        verdict, msg = evaluate_autosomal_or_female_verdict(m_fail, min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("(high)", msg)

        m_low_refs = {
            "n_valid": 1000,
            "std_of_means": 1.0,
            "outlier_pct": 0.0,
            "n_low_refs": 6,
        }
        verdict, msg = evaluate_autosomal_or_female_verdict(
            m_low_refs, min_ref_bins=150
        )
        self.assertEqual(verdict, "WARN")
        self.assertIn("n_refs<150", msg)

    def test_evaluate_male_verdict_heuristics(self) -> None:
        m_pass = {
            "n_valid": 1000,
            "mean_of_means": 1.0,
            "outlier_pct": 0.0,
            "n_low_refs": 0,
            "signal_contig_metrics": {
                "signal_contig": "chrY",
                "n_valid": 100,
                "usable_pct": 80.0,
                "mean_of_means": 2.0,
            },
        }
        verdict, msg = evaluate_male_verdict(m_pass, min_ref_bins=150)
        self.assertEqual(verdict, "PASS")

        # Warn on usable_pct < 50
        m_chry_warn = dict(m_pass)
        m_chry_warn["signal_contig_metrics"] = {
            "signal_contig": "chrY",
            "n_valid": 100,
            "usable_pct": CHRY_MIN_USABLE_PCT_WARN - 5.0,
            "mean_of_means": 2.0,
        }
        verdict, msg = evaluate_male_verdict(m_chry_warn, min_ref_bins=150)
        self.assertEqual(verdict, "WARN")
        self.assertIn("usable chrY bins", msg)

        # Fail on usable_pct < 20
        m_chry_fail = dict(m_pass)
        m_chry_fail["signal_contig_metrics"] = {
            "signal_contig": "chrY",
            "n_valid": 100,
            "usable_pct": CHRY_MIN_USABLE_PCT_FAIL - 5.0,
            "mean_of_means": 2.0,
        }
        verdict, msg = evaluate_male_verdict(m_chry_fail, min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("usable chrY bins", msg)

        # High mean distance on chrY
        m_chry_dist_fail = dict(m_pass)
        m_chry_dist_fail["signal_contig_metrics"] = {
            "signal_contig": "chrY",
            "n_valid": 100,
            "usable_pct": 90.0,
            "mean_of_means": CHRY_MEAN_DIST_FAIL + 10.0,
        }
        verdict, msg = evaluate_male_verdict(m_chry_dist_fail, min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("very poor chrY", msg)

    def test_run_reference_qc_on_synthetic_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            n_bins = 20
            table = make_reference_table(
                branches=["autosomal"] * n_bins,
                contigs=["chr1"] * n_bins,
                starts=[i * 100 for i in range(n_bins)],
                ends=[(i + 1) * 100 for i in range(n_bins)],
                contig_lengths={"chr1": 2000},
                gc=[0.42] * n_bins,
                mappability=[0.95] * n_bins,
                pca_mean=[1.0] * n_bins,
                pca_loadings=[[0.1, 0.2]] * n_bins,
                mean_distance=[0.55] * n_bins,
                null_mean=[0.0] * n_bins,
                null_std=[0.1] * n_bins,
                sample_identities=[{"path": "s1.ccnv"}, {"path": "s2.ccnv"}],
                model={
                    "bin_size": 100,
                    "branches": ["autosomal"],
                    "autosomal_contigs": ["chr1"],
                    "sex_contigs": [],
                    "sex_signal_contig": None,
                    "sex_cutoff": None,
                    "sex_source": "none",
                    "sample_sexes": None,
                    "sex_inference": None,
                    "pca_components": 2,
                    "nearest_bins": 5,
                    "max_batch_memory_mb": 64,
                    "flat": False,
                },
            )

            outputs = run_reference_qc_on_table(
                table,
                prefix=tmp_path / "ref",
                write_json=True,
                emit_stderr=False,
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertEqual(outputs.worst_severity, 0)
            self.assertTrue(outputs.json_path.is_file())

            with open(outputs.json_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["overall_verdict"], "PASS")
            self.assertIn("autosomal", saved["branches"])
            self.assertEqual(saved["cohort"]["sample_count"], 2)

    def test_run_reference_qc_flat_fasta_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            n_bins = 10
            table = make_reference_table(
                branches=["autosomal"] * n_bins,
                contigs=["chr1"] * n_bins,
                starts=[i * 100 for i in range(n_bins)],
                ends=[(i + 1) * 100 for i in range(n_bins)],
                contig_lengths={"chr1": 1000},
                gc=[0.42] * n_bins,
                mappability=[0.95] * n_bins,
                pca_mean=[1.0] * n_bins,
                pca_loadings=[[0.0, 0.0]] * n_bins,
                mean_distance=[0.15] * n_bins,
                null_mean=[0.0] * n_bins,
                null_std=[0.1] * n_bins,
                sample_identities=[],
                model={
                    "bin_size": 100,
                    "branches": ["autosomal"],
                    "autosomal_contigs": ["chr1"],
                    "sex_contigs": [],
                    "sex_signal_contig": None,
                    "sex_cutoff": None,
                    "sex_source": "none",
                    "sample_sexes": None,
                    "sex_inference": None,
                    "pca_components": 2,
                    "nearest_bins": 5,
                    "max_batch_memory_mb": 64,
                    "flat": True,
                },
            )

            outputs = run_reference_qc_on_table(
                table,
                prefix=tmp_path / "flat_ref",
                write_json=True,
                emit_stderr=False,
            )
            self.assertEqual(outputs.overall_verdict, "WARN")
            self.assertEqual(outputs.worst_severity, 1)
            self.assertTrue(
                any("flat FASTA" in msg for msg in outputs.data["messages"])
            )

    def test_run_reference_qc_file_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            nonexistent = tmp_path / "does_not_exist.ccnv"
            with self.assertRaises(ReferenceQCError):
                run_reference_qc(ReferenceQCConfig(reference=nonexistent))

    def test_evaluate_reference_qc_prenatal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            branches = (
                ["autosomal"] * 10 + ["prenatal_autosomal"] * 2 + ["prenatal_ffy"] * 20
            )
            contigs = ["chr1"] * 10 + ["chr1"] * 2 + ["chrY"] * 20
            starts = (
                [i * 5000 for i in range(10)]
                + [0, 100000]
                + [i * 5000 for i in range(20)]
            )
            ends = (
                [(i + 1) * 5000 for i in range(10)]
                + [100000, 200000]
                + [(i + 1) * 5000 for i in range(20)]
            )
            n_total = len(branches)
            masked = [False] * 10 + [False, True] + [False] * 15 + [True] * 5

            table = make_reference_table(
                branches=branches,
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths={"chr1": 200000, "chrY": 100000},
                masked=masked,
                gc=[0.42] * n_total,
                mappability=[0.95] * n_total,
                pca_mean=[1.0] * n_total,
                pca_loadings=[[0.1, 0.2]] * n_total,
                mean_distance=[0.55] * n_total,
                null_mean=[0.0] * n_total,
                null_std=[0.1] * n_total,
                sample_identities=[{"path": "s1.ccnv"}, {"path": "s2.ccnv"}],
                model={
                    "bin_size": 5000,
                    "prenatal_bin_size": 100000,
                    "mode": "prenatal",
                    "branches": [
                        "autosomal",
                        "prenatal_autosomal",
                        "prenatal_ffy",
                    ],
                    "autosomal_contigs": ["chr1"],
                    "sex_contigs": ["chrY"],
                    "sex_signal_contig": None,
                    "sex_cutoff": None,
                    "sex_source": "prenatal_model",
                    "sample_sexes": ["female", "male"],
                    "sex_inference": None,
                    "pca_components": 2,
                    "nearest_bins": 5,
                    "max_batch_memory_mb": 64,
                    "flat": False,
                },
                extra_metadata={
                    "prenatal": {
                        "prenatal_bin_size": 100000,
                        "yfrac_threshold": 0.005,
                        "ffy_intercept": 0.0,
                        "ffy_slope": 1.0,
                        "olm_weights": [0.1],
                        "olm_bias": 0.0,
                        "ffx_intercept": 2.0,
                        "ffx_slope": -2.0,
                        "female_chrx_ratio_mean": 1.0,
                        "female_chrx_ratio_std": 0.015,
                    },
                },
            )

            outputs = run_reference_qc_on_table(
                table,
                prefix=tmp_path / "prenatal_qc_ref",
                write_json=True,
                emit_stderr=False,
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertIn("prenatal", outputs.data)
            p_qc = outputs.data["prenatal"]
            self.assertEqual(p_qc["usable_autosomal_bins"], 1)  # 1 unmasked
            self.assertEqual(p_qc["informative_ffy_bins"], 15)  # 15 unmasked
            self.assertAlmostEqual(p_qc["yfrac_threshold"], 0.005)
            self.assertAlmostEqual(p_qc["female_chrx_ratio_std"], 0.015)

            # Branches should only contain CNV branches
            self.assertIn("autosomal", outputs.data["branches"])
            self.assertNotIn("prenatal_autosomal", outputs.data["branches"])
            self.assertNotIn("prenatal_ffy", outputs.data["branches"])


if __name__ == "__main__":
    unittest.main()
