"""Tests for direct BAM/CRAM ingestion and segment origin calling in predict."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import make_sample_table
from published_fixtures import (
    PUBLISHED_CRAM,
    ensure_published_fasta,
    extract_published_samples,
)
from scipy.stats import norm

from ccnv.features.fragmentomics import resolve_fragmentomics_compatibility
from ccnv.features.prenatal import annotate_prenatal_segments
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet
from ccnv.models import (
    PredictConfig,
    PredictError,
    ReferenceConfig,
    SegmentResult,
)
from ccnv.models.results import FetalFractionResult, SCAResult
from ccnv.models.schemas import (
    BRANCH_AUTOSOMAL,
    BRANCH_PRENATAL_AUTOSOMAL,
    BRANCH_PRENATAL_FFY,
    FRAGMENTOMICS_METADATA_KEY,
    REFERENCE_SCHEMA,
    ArtifactKind,
    FragmentomicsMetadata,
    PrenatalModelMetadata,
    ReferenceMetadata,
    metadata_for,
)
from ccnv.predict import (
    PredictionInput,
    build_branch_model,
    load_inputs,
)
from ccnv.predict import (
    run as run_predict,
)
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
                call="gain",
            ),
            SegmentResult(
                contig="chr1",
                start=10000,
                end=20000,
                ratio=0.07,
                zscore=4.5,
                weight=1.0,
                bin_count=1,
                call="gain",
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

    def test_fragmentomics_compatibility_and_observed_qc_counts(self) -> None:
        # 1. ReferenceMetadata round-trip persists fragmentomics_corrected
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
            counts=[100, 120],
            contigs=["chr1", "chr1"],
            starts=[0, 10000],
            ends=[10000, 20000],
            contig_lengths={"chr1": 20000},
            bin_size=10000,
            endpoint_counts=[50, 60],
        )
        pe_frag_meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=220,
            chromatosome_mode_bp=166.0,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.35,
            dinucleosome_fraction=0.05,
            high_mw_fraction=0.01,
            helical_10bp_amplitude=0.04,
            helical_10bp_phase=1.0,
            motif_diversity_score=0.90,
            hist_autosomal=[1] * 571,
            hist_chry=[0] * 571,
            hist_chrx=[0] * 571,
            end_motif_counts=[1] * 256,
        )
        pe_sample = make_sample_table(
            counts=[100, 120],
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
            return pa.table(
                {
                    "branch": [BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
                    "contig": ["chr1", "chr1"],
                    "start": [0, 10000],
                    "end": [10000, 20000],
                    "masked": [False, False],
                    "gc": [-1.0, -1.0],
                    "mappability": [-1.0, -1.0],
                    "pca_mean": [100.0, 120.0],
                    "pca_loadings": [[], []],
                    "nearest_ref_indices": [[1], [0]],
                    "mean_distance": [0.5, 0.5],
                    "null_mean": [0.0, 0.0],
                    "null_std": [0.05, 0.05],
                },
                schema=REFERENCE_SCHEMA.with_metadata(
                    metadata_for(
                        ArtifactKind.REFERENCE,
                        contig_lengths={"chr1": 20000},
                        values={"sample_identities": ["s1"], "model": model_dict},
                    )
                ),
            )

        legacy_ref = _make_ref({"bin_size": 10000})
        se_ref = _make_ref({"bin_size": 10000, "fragmentomics_corrected": False})
        pe_ref = _make_ref({"bin_size": 10000, "fragmentomics_corrected": True})
        flat_ref = _make_ref({"bin_size": 10000, "flat": True})

        # Legacy & single-end references accept single-end (skipping endpoint_ratio) and reject paired-end
        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, legacy_ref))
        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, se_ref))
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(pe_sample, legacy_ref)
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(pe_sample, se_ref)

        # Paired-end reference accepts paired-end and rejects single-end
        self.assertTrue(resolve_fragmentomics_compatibility(pe_sample, pe_ref))
        with self.assertRaises(PredictError):
            resolve_fragmentomics_compatibility(se_sample, pe_ref)

        # Flat FASTA reference explicitly supports both
        self.assertFalse(resolve_fragmentomics_compatibility(se_sample, flat_ref))
        self.assertTrue(resolve_fragmentomics_compatibility(pe_sample, flat_ref))

        # Single-end build_branch_model leaves mean_frag_len & endpoint_ratio as None
        se_bm = build_branch_model(
            PredictionInput(
                sample=se_sample,
                reference=se_ref,
                contig_lengths={"chr1": 20000},
                sex=None,
                branches=(BRANCH_AUTOSOMAL,),
            ),
            BRANCH_AUTOSOMAL,
        )
        self.assertIsNone(se_bm.mean_frag_len)
        self.assertIsNone(se_bm.endpoint_ratio)

        # 2. Prenatal prediction QC preserves observed (unweighted) sample counts
        grid = np.arange(30, 601, dtype=np.float64)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_fet /= np.sum(p_fet)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_mat /= np.sum(p_mat)
        prenatal_meta = PrenatalModelMetadata(
            prenatal_bin_size=20000,
            yfrac_threshold=0.005,
            ffy_intercept=0.0,
            ffy_slope=1.0,
            mlp_w1=[[0.0]],
            mlp_b1=[0.0],
            mlp_w2=[[0.0]],
            mlp_b2=[-2.0],
            mlp_skew_intercept=0.0,
            mlp_skew_slope=1.0,
            olm_weights=[0.0],
            olm_bias=-2.0,
            olm_skew_intercept=0.0,
            olm_skew_slope=1.0,
            ffx_intercept=2.0,
            ffx_slope=-2.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            p_fetal_density=p_fet.tolist(),
            p_maternal_density=p_mat.tolist(),
        )
        prenatal_rows = pa.table(
            {
                "branch": [
                    BRANCH_AUTOSOMAL,
                    BRANCH_AUTOSOMAL,
                    BRANCH_PRENATAL_AUTOSOMAL,
                    BRANCH_PRENATAL_FFY,
                    BRANCH_PRENATAL_FFY,
                ],
                "contig": ["chr1", "chr1", "chr1", "chrY", "chrY"],
                "start": [0, 10000, 0, 0, 10000],
                "end": [10000, 20000, 20000, 10000, 20000],
                "masked": [False, False, False, False, False],
                "gc": [-1.0] * 5,
                "mappability": [-1.0] * 5,
                "pca_mean": [100.0, 120.0, 220.0, 1.0, 1.0],
                "pca_loadings": [[], [], [1.0], [0.5], [0.5]],
                "nearest_ref_indices": [[1], [0], [], [], []],
                "mean_distance": [0.5, 0.5, 0.0, 0.0, 0.0],
                "null_mean": [0.0] * 5,
                "null_std": [0.05, 0.05, 0.0, 0.0, 0.0],
            },
            schema=REFERENCE_SCHEMA.with_metadata(
                metadata_for(
                    ArtifactKind.REFERENCE,
                    contig_lengths={"chr1": 20000, "chrY": 20000},
                    values={
                        "sample_identities": ["s1"],
                        "model": {
                            "bin_size": 10000,
                            "mode": "prenatal",
                            "prenatal_bin_size": 20000,
                            "fragmentomics_corrected": True,
                        },
                        "prenatal": prenatal_meta.to_dict(),
                    },
                )
            ),
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
            assert outputs.qc_json_path is not None
            qc_payload = json.loads(outputs.qc_json_path.read_text(encoding="utf-8"))
            # Observed counts are [100, 120, 0, 0] -> mean 55.0, median 50.0 (not inflated by ~2.5x Fisher weights!)
            self.assertAlmostEqual(
                qc_payload["read_metrics"]["mean_bin_count"], 55.0, places=4
            )
            self.assertAlmostEqual(
                qc_payload["read_metrics"]["median_bin_count"], 50.0, places=4
            )


if __name__ == "__main__":
    unittest.main()
