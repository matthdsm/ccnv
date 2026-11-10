"""Tests for direct BAM/CRAM ingestion and segment origin calling in predict."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_sample_table
from published_fixtures import (
    PUBLISHED_CRAM,
    ensure_published_fasta,
    extract_published_samples,
)

from ccnv.features.prenatal import annotate_prenatal_segments
from ccnv.models import (
    ReferenceConfig,
    SegmentResult,
)
from ccnv.models.results import FetalFractionResult, SCAResult
from ccnv.models.schemas import (
    PrenatalModelMetadata,
)
from ccnv.predict import load_inputs
from ccnv.reference import run as run_reference


class TestPredictPolymorphicAndOrigin(unittest.TestCase):
    def test_load_inputs_polymorphic_cram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s1, s2 = extract_published_samples(root)
            ref_path = root / "cohort_ref.ccnv"
            run_reference(
                ReferenceConfig(
                    samples=(s1, s2),
                    prefix=root / "cohort_ref",
                    bin_size=100000,
                    min_cohort_size=2,
                )
            )

            fasta = ensure_published_fasta()
            # Direct CRAM path passed as sample
            inputs = load_inputs(
                PUBLISHED_CRAM,
                ref_path,
                sex="autosomal",
                mode="germline",
                fasta=fasta,
            )
            self.assertIsNotNone(inputs.sample)
            self.assertIn("fragment_lengths", inputs.sample.column_names)
            self.assertGreater(inputs.sample.num_rows, 0)

    def test_annotate_prenatal_segments_origin_fetal_and_maternal(self) -> None:
        grid_len = 571
        p_fet = np.zeros(grid_len, dtype=np.float64)
        p_mat = np.zeros(grid_len, dtype=np.float64)
        # 143 bp -> index 143 - 30 = 113
        p_fet[113] = 1.0
        # 166 bp -> index 166 - 30 = 136
        p_mat[136] = 1.0

        prenatal_meta = PrenatalModelMetadata(
            prenatal_bin_size=100000,
            yfrac_threshold=0.001,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            mlp_w1=[],
            mlp_b1=[],
            mlp_w2=[],
            mlp_b2=[],
            mlp_skew_intercept=0.0,
            mlp_skew_slope=1.0,
            olm_weights=[],
            olm_bias=0.0,
            olm_skew_intercept=0.0,
            olm_skew_slope=1.0,
            ffx_intercept=2.0,
            ffx_slope=-2.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            ffy_std=0.01,
            ffa_std=0.02,
            ff_size_std=0.018,
            p_fetal_density=[float(x) for x in p_fet],
            p_maternal_density=[float(x) for x in p_mat],
        )

        prenatal_res = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.10,
            active_ff_source="FFY",
            ffy=0.10,
            ffx=0.10,
            ffa=0.10,
            chrx_ratio=0.95,
            chry_score=0.01,
            yfrac_threshold=0.001,
            sca=SCAResult("XY", False, 1.0, 0.0),
        )

        # Segment 1: Fetal-enriched (lengths=143 bp)
        # Segment 2: Maternal-enriched (lengths=166 bp)
        sample = make_sample_table(
            counts=[100, 100],
            contigs=["chr1", "chr1"],
            starts=[0, 10000],
            ends=[10000, 20000],
            bin_size=10000,
            fragment_lengths=[[143] * 50, [166] * 50],
        )

        segs = [
            SegmentResult(
                contig="chr1",
                start=0,
                end=10000,
                ratio=0.07,
                zscore=4.5,
                weight=1.0,
                bin_count=1,
                call="GAIN",
            ),
            SegmentResult(
                contig="chr1",
                start=10000,
                end=20000,
                ratio=0.07,
                zscore=4.5,
                weight=1.0,
                bin_count=1,
                call="GAIN",
            ),
        ]

        annotated = annotate_prenatal_segments(
            segs,
            prenatal_res,
            sample_table=sample,
            prenatal_meta=prenatal_meta,
        )

        self.assertEqual(len(annotated), 2)
        # First segment should be fetal confirmed
        self.assertGreater(annotated[0].frag_llr_size, 2.0)
        self.assertEqual(annotated[0].origin_call, "FETAL_CONFIRMED")
        # Second segment should be maternal confirmed
        self.assertLess(annotated[1].frag_llr_size, -2.0)
        self.assertEqual(annotated[1].origin_call, "MATERNAL_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
