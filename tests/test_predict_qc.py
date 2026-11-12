"""Unit tests for predict QC payload generation and serialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
from helpers import metadata_for

from ccnv.models import (
    NUM_FRAG_BANDS,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    ReadFilters,
)
from ccnv.models.results import FetalFractionResult, SCAResult, SegmentResult
from ccnv.qc.predict import build_predict_qc_payload, write_predict_qc_json


def _make_sample_table() -> pa.Table:
    contig_lengths = {"chr1": 200, "chr2": 100, "chrX": 100}
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": 100,
            "contig_order": ["chr1", "chr2", "chrX"],
            "source_identity": {"path": "/data/test_sample.bam", "format": "bam"},
            "reference_identity": {"id": "genome"},
            "qc": {
                "effective_reads_examined": 1000,
                "effective_reads_kept": 850,
                "autosomal_reads_kept": 800,
                "sex_reads_kept": 50,
                "filter_unmapped": 50,
                "filter_mapq": 40,
                "filter_duplicate": 30,
                "filter_secondary": 10,
                "filter_supplementary": 10,
                "filter_improper_pair": 5,
                "filter_no_start": 5,
            },
            "filters": asdict(ReadFilters()),
        },
    )
    return pa.table(
        {
            "count": pa.array([250, 250, 300, 50], type=pa.uint32()),
            "paired_count": pa.array([0, 0, 0, 0], type=pa.uint32()),
            "frag_len_sum": pa.array([0, 0, 0, 0], type=pa.uint64()),
            "short_frag_count": pa.array([0, 0, 0, 0], type=pa.uint32()),
            "endpoint_count": pa.array([0, 0, 0, 0], type=pa.uint32()),
            "frag_gc_sum": pa.array([0.0, 0.0, 0.0, 0.0], type=pa.float32()),
            "frag_bands": pa.nulls(4, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


class TestPredictQC(unittest.TestCase):
    def test_build_predict_qc_payload_germline(self) -> None:
        table = _make_sample_table()
        segments = (
            SegmentResult("chr1", 0, 100, 0.4, 3.5, 1.0, 1, "gain"),
            SegmentResult("chr1", 100, 200, -0.5, -3.2, 1.0, 1, "loss"),
            SegmentResult("chr2", 0, 100, 0.0, 0.1, 1.0, 1, None),
        )
        payload = build_predict_qc_payload("germline", table, segments)

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["mode"], "germline")
        self.assertEqual(payload["sample"], "test_sample")
        self.assertIsNone(payload["prenatal"])

        read_metrics = payload["read_metrics"]
        assert isinstance(read_metrics, dict)
        self.assertEqual(read_metrics["effective_reads_examined"], 1000)
        self.assertEqual(read_metrics["effective_reads_kept"], 850)
        self.assertEqual(read_metrics["autosomal_reads_kept"], 800)
        self.assertEqual(read_metrics["sex_reads_kept"], 50)
        self.assertEqual(read_metrics["total_bins"], 4)
        self.assertEqual(read_metrics["usable_bins"], 4)
        self.assertEqual(read_metrics["masked_bins"], 0)
        self.assertAlmostEqual(read_metrics["mean_bin_count"], 212.5)

        filter_drops = read_metrics["filter_drops"]
        assert isinstance(filter_drops, dict)
        self.assertEqual(filter_drops["unmapped"], 50)
        self.assertEqual(filter_drops["duplicate"], 30)

        cnv_summary = payload["cnv_summary"]
        assert isinstance(cnv_summary, dict)
        self.assertEqual(cnv_summary["total_segments"], 3)
        self.assertEqual(cnv_summary["gain_calls"], 1)
        self.assertEqual(cnv_summary["loss_calls"], 1)
        self.assertEqual(cnv_summary["discordant_segments"], 0)

    def test_build_predict_qc_payload_prenatal(self) -> None:
        table = _make_sample_table()
        sca = SCAResult(
            call="XY",
            is_aneuploid=False,
            confidence=0.98,
            chrx_zscore=-0.2,
            ff_delta=0.005,
        )
        prenatal_result = FetalFractionResult(
            fetal_sex="M",
            active_ff=0.12,
            active_ff_source="FFY",
            ffy=0.12,
            ffx=0.115,
            ffa=0.118,
            ff_size=0.122,
            chrx_ratio=0.94,
            chry_score=0.006,
            yfrac_threshold=0.004,
            sca=sca,
            low_ff_warning=False,
            warnings=(),
        )
        segments = (
            SegmentResult(
                "chr1",
                0,
                100,
                0.08,
                3.1,
                1.0,
                1,
                "gain",
                expected_ratio=0.084,
                implied_ff=0.115,
                discordant=False,
            ),
            SegmentResult(
                "chr2",
                0,
                100,
                0.01,
                3.1,
                1.0,
                1,
                "gain",
                expected_ratio=0.084,
                implied_ff=0.014,
                discordant=True,
                discordance_reason="SUB_FETAL_RATIO",
            ),
        )
        payload = build_predict_qc_payload(
            "prenatal", table, segments, prenatal_result=prenatal_result
        )

        self.assertEqual(payload["mode"], "prenatal")
        cnv_summary = payload["cnv_summary"]
        assert isinstance(cnv_summary, dict)
        self.assertEqual(cnv_summary["discordant_segments"], 1)

        prenatal = payload["prenatal"]
        assert isinstance(prenatal, dict)
        self.assertEqual(prenatal["fetal_sex"], "M")
        self.assertEqual(prenatal["active_ff"], 0.12)
        self.assertEqual(prenatal["active_ff_source"], "FFY")
        self.assertEqual(prenatal["ffy"], 0.12)
        self.assertEqual(prenatal["ffx"], 0.115)
        self.assertEqual(prenatal["ffa"], 0.118)
        self.assertEqual(prenatal["ff_size"], 0.122)
        self.assertFalse(prenatal["low_ff_warning"])

        sca_dict = prenatal["sca"]
        assert isinstance(sca_dict, dict)
        self.assertEqual(sca_dict["call"], "XY")
        self.assertFalse(sca_dict["is_aneuploid"])
        self.assertEqual(sca_dict["confidence"], 0.98)

    def test_write_predict_qc_json(self) -> None:
        table = _make_sample_table()
        payload = build_predict_qc_payload("germline", table, ())

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_qc.json"
            out = write_predict_qc_json(json_path, payload)
            self.assertEqual(out, json_path)
            self.assertTrue(json_path.is_file())

            with open(json_path, encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["mode"], "germline")
            self.assertEqual(loaded["sample"], "test_sample")
            self.assertIn("read_metrics", loaded)
            self.assertIn("cnv_summary", loaded)


if __name__ == "__main__":
    unittest.main()
