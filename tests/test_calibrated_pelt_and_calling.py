"""Contract tests for calibrated PELT segmentation, dyadic BLUE, Bayes factors, and Ref-QC."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_reference_table

from ccnv.models import BRANCH_MALE_SEX
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.stats.segmentation import (
    ContigSegmentationInput,
    estimate_dyadic_inflation,
    segment_and_score_genome,
)


class TestCalibratedPeltAndCalling(unittest.TestCase):
    """Test suite for calibrated PELT, dyadic BLUE scoring, CN Bayes factors, and Ref-QC."""

    def test_homozygous_cn0_deletion_and_interior_focal_50kb_deletion_detected(
        self,
    ) -> None:
        """Verify 20-bin CN=0 deletion at 25 Mb and 10-bin CN=1 deletion at 50 Mb are detected."""
        n_bins = 20_000
        bin_size = 5_000
        contig_len = n_bins * bin_size  # 100 Mb
        starts = np.arange(0, contig_len, bin_size, dtype=np.int64)
        ends = starts + bin_size

        ratios = np.zeros(n_bins, dtype=np.float64)
        weights = np.full(n_bins, 4.0, dtype=np.float64)
        zscores = np.zeros(n_bins, dtype=np.float64)
        counts = np.full(n_bins, 8.0, dtype=np.float64)
        expected = np.full(n_bins, 8.0, dtype=np.float64)

        # 20-bin CN=0 deletion at 25 Mb (bins 5000 to 5020)
        ratios[5000:5020] = -4.5
        zscores[5000:5020] = -15.0
        counts[5000:5020] = 0.0

        # 10-bin CN=1 deletion at 50 Mb (bins 10000 to 10010)
        ratios[10000:10010] = -1.0
        zscores[10000:10010] = -5.0
        counts[10000:10010] = 4.0

        arm_inputs = {
            "chr1": ContigSegmentationInput(
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=contig_len,
                zscores=zscores,
                counts=counts,
                expected=expected,
            )
        }
        segs = segment_and_score_genome(
            arm_inputs,
            sensitivity=1.0,
            mode="germline",
            z_threshold=3.0,
        )
        calls = [s for s in segs if s.call == "loss"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].copy_number, 0)
        self.assertEqual(calls[1].copy_number, 1)

    def test_dyadic_wave_inflation_suppresses_megabase_wave_false_positives_at_1kb(
        self,
    ) -> None:
        """Verify scale-dependent variance inflation kappa(1024) captures 1 Mb megabase waves."""
        rng = np.random.default_rng(42)
        n_bins = 50_000
        noise = rng.normal(0.0, np.sqrt(0.98), size=n_bins)
        wave = np.sqrt(2.0 * 0.02) * np.sin(2.0 * np.pi * np.arange(n_bins) / 5000.0)
        wavy_z = noise + wave
        usable_all = np.ones(n_bins, dtype=bool)

        infl = estimate_dyadic_inflation(wavy_z, usable_all)
        self.assertGreater(float(infl[-1]), 8.0)

    def test_germline_integer_cn_bayes_factor_rejects_cn_203_high_depth_ripple(
        self,
    ) -> None:
        """Verify discrete CN Bayes factor rejects sub-copy ripple even with Z_BLUE = 4.5."""
        n_bins = 100
        bin_size = 500_000
        contig_len = n_bins * bin_size
        starts = np.arange(0, contig_len, bin_size, dtype=np.int64)
        ends = starts + bin_size

        ratios = np.zeros(n_bins, dtype=np.float64)
        weights = np.full(n_bins, 4.0, dtype=np.float64)
        zscores = np.zeros(n_bins, dtype=np.float64)
        counts = np.full(n_bins, 1600.0, dtype=np.float64)
        expected = np.full(n_bins, 1600.0, dtype=np.float64)

        # 40-bin segment with +0.022 log2 ratio (CN = 2.03) and Z_BLUE = 4.5
        ratios[30:70] = 0.022
        zscores[30:70] = 4.5 / np.sqrt(40.0)
        counts[30:70] = 1600.0 * 1.015355

        high_depth_ripple_inputs = {
            "chr1": ContigSegmentationInput(
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=contig_len,
                zscores=zscores,
                counts=counts,
                expected=expected,
            )
        }
        segs = segment_and_score_genome(
            high_depth_ripple_inputs,
            mode="germline",
            z_threshold=3.0,
        )
        self.assertTrue(all(s.call is None for s in segs))

    def test_ref_qc_fails_male_branch_when_100pct_chry_bins_masked(self) -> None:
        """Verify reference QC fails male_sex branch when 100% of chrY bins are masked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            n_auto = 20
            n_chry = 10
            total_bins = n_auto + n_chry

            branches = [BRANCH_MALE_SEX] * total_bins
            contigs = ["chr1"] * n_auto + ["chrY"] * n_chry
            starts = [i * 100 for i in range(n_auto)] + [i * 100 for i in range(n_chry)]
            ends = [(i + 1) * 100 for i in range(n_auto)] + [
                (i + 1) * 100 for i in range(n_chry)
            ]
            masked = [False] * n_auto + [True] * n_chry

            table = make_reference_table(
                branches=branches,
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths={"chr1": n_auto * 100, "chrY": n_chry * 100},
                masked=masked,
                gc=[0.42] * total_bins,
                mappability=[0.95] * total_bins,
                pca_mean=[1.0] * total_bins,
                pca_loadings=[[0.1, 0.2]] * total_bins,
                mean_distance=[0.55] * total_bins,
                null_mean=[0.0] * total_bins,
                bio_dispersion=[0.05] * total_bins,
                model={
                    "bin_size": 100,
                    "branches": [BRANCH_MALE_SEX],
                    "autosomal_contigs": ["chr1"],
                    "sex_contigs": ["chrY"],
                    "sex_signal_contig": "chrY",
                    "sex_cutoff": 0.05,
                    "sex_source": "inferred",
                    "pca_components": 2,
                    "max_batch_memory_mb": 64,
                    "flat": False,
                },
                sample_identities=[{"path": "s1.ccnv"}, {"path": "s2.ccnv"}],
            )

            verdict = run_reference_qc_on_table(
                table,
                prefix=tmp_path / "ref",
                write_json=False,
                emit_stderr=False,
            )
            self.assertEqual(verdict.branches[BRANCH_MALE_SEX].status, "FAIL")
