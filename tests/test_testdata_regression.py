"""Functional and regression tests covering all synthetic and real-world testdata surfaces."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import published_fixtures as pf
import pyarrow.parquet as pq

from ccnv.features.cohort import load_and_assemble_cohort
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import read_reference_parquet, read_sample_buffers
from ccnv.io.samplesheet import load_samplesheet
from ccnv.io.tracks import load_cytobands_track, load_gc_track, load_mappability_track
from ccnv.models import (
    AnalysisMode,
    PredictConfig,
    ReferenceConfig,
    ReferenceError,
    SamplesheetEntry,
    SchemaVersionError,
    validate_sample_table,
)
from ccnv.predict import run as run_predict
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.reference import run as run_reference

_RUN_REAL_TESTDATA = os.environ.get("CCNV_RUN_REAL_TESTDATA") == "1"

_SYNTHETIC_COHORT_SAMPLES: tuple[str, ...] = (
    "reference/synthetic_cohort/samples/ctrl_f01.ccnv",
    "reference/synthetic_cohort/samples/ctrl_f02.ccnv",
    "reference/synthetic_cohort/samples/ctrl_f03.ccnv",
    "reference/synthetic_cohort/samples/ctrl_f04.ccnv",
    "reference/synthetic_cohort/samples/ctrl_f05.ccnv",
    "reference/synthetic_cohort/samples/ctrl_f06.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m01.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m02.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m03.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m04.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m05.ccnv",
    "reference/synthetic_cohort/samples/ctrl_m06.ccnv",
)


class TestSyntheticTestdataRegression(unittest.TestCase):
    """Fast deterministic tests exercising every synthetic fixture in copynumber-cnv-testdata."""

    def test_synthetic_tracks_load_and_agree_between_bigwig_and_bed(self) -> None:
        truth = pf.load_testdata_truth("reference/synthetic_cohort/truth.json")
        contig_lengths = truth["contig_lengths"]
        if not isinstance(contig_lengths, dict):
            self.fail("expected contig_lengths dict in reference truth.json")
        bin_size = (
            int(truth["bin_size"]) if isinstance(truth["bin_size"], int) else 5000
        )

        contigs: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for name, length in contig_lengths.items():
            for s in range(0, int(length), bin_size):
                contigs.append(str(name))
                starts.append(s)
                ends.append(min(s + bin_size, int(length)))
        s_arr = np.asarray(starts, dtype=np.int64)
        e_arr = np.asarray(ends, dtype=np.int64)

        gc_bw = pf.ensure_testdata_file("tracks/synthetic_cohort_gc.bw")
        gc_bed = pf.ensure_testdata_file("tracks/synthetic_cohort_gc.bed.gz")
        map_bw = pf.ensure_testdata_file("tracks/synthetic_cohort_mappability.bw")
        cyto_bed = pf.ensure_testdata_file("tracks/synthetic_cohort_cytobands.bed.gz")

        vals_bw, _, _ = load_gc_track(gc_bw, contigs, s_arr, e_arr)
        vals_bed, _, _ = load_gc_track(gc_bed, contigs, s_arr, e_arr)
        np.testing.assert_allclose(vals_bw, vals_bed, atol=1e-4)

        map_vals = load_mappability_track(map_bw, contigs, s_arr, e_arr)
        self.assertTrue(bool(np.any(map_vals < 0.5)))
        self.assertTrue(bool(np.any(map_vals >= 0.9)))

        bands = load_cytobands_track(
            cyto_bed, {str(k): int(v) for k, v in contig_lengths.items()}
        )
        self.assertEqual(len(bands), len(contig_lengths) * 4)

    def test_synthetic_cohort_samplesheets_rebinning_and_negative_contracts(
        self,
    ) -> None:
        for rel in _SYNTHETIC_COHORT_SAMPLES:
            pf.ensure_testdata_file(rel)
        ss_yaml = pf.ensure_testdata_file(
            "reference/synthetic_cohort/samplesheet_germline.yaml"
        )
        ss_json = pf.ensure_testdata_file(
            "reference/synthetic_cohort/samplesheet_germline.json"
        )
        gc_bw = pf.ensure_testdata_file("tracks/synthetic_cohort_gc.bw")
        map_bw = pf.ensure_testdata_file("tracks/synthetic_cohort_mappability.bw")

        yaml_entries = load_samplesheet(ss_yaml)
        json_entries = load_samplesheet(ss_json)
        self.assertEqual(len(yaml_entries), 12)
        self.assertEqual(len(json_entries), 12)
        self.assertTrue(all(e.sex is None for e in yaml_entries))
        self.assertEqual([e.sex for e in json_entries], ["F"] * 6 + ["M"] * 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = run_reference(
                ReferenceConfig(
                    prefix=Path(tmpdir) / "from_json",
                    samplesheet_path=ss_json,
                    bin_size=5_000,
                    gc=gc_bw,
                    mappability=map_bw,
                    min_cohort_size=10,
                    min_sex_group_size=5,
                    qc_report=False,
                )
            )
            _, ref_meta = read_reference_parquet(ref_path, check_rows=True)
            self.assertEqual(ref_meta.sample_count, 12)
            self.assertEqual(ref_meta.sex_source, "samplesheet")

        se_ctrl = pf.ensure_testdata_file(
            "reference/synthetic_cohort/single_end_ctrl.ccnv"
        )
        pe_ctrl = pf.ensure_testdata_file(_SYNTHETIC_COHORT_SAMPLES[0])
        with self.assertRaisesRegex(ReferenceError, "mixed single-end and paired-end"):
            load_and_assemble_cohort(
                (SamplesheetEntry(pe_ctrl), SamplesheetEntry(se_ctrl)),
                target_bin_size=5_000,
            )

        non_nested = pf.ensure_testdata_file(
            "reference/synthetic_cohort/non_nested_7kb_sample.ccnv"
        )
        bufs_7k = read_sample_buffers(non_nested)
        bufs_10k = rebin_sample_buffers(bufs_7k, 10_000)
        self.assertEqual(bufs_10k.meta.bin_size, 10_000)
        self.assertIsNone(bufs_10k.meta.fragmentomics)
        self.assertEqual(int(np.sum(bufs_10k.paired_count)), 0)
        self.assertAlmostEqual(
            float(np.sum(bufs_10k.count)), float(np.sum(bufs_7k.count)), delta=2.0
        )

        v1_sample = pf.ensure_testdata_file(
            "reference/synthetic_cohort/incompatible_v1_sample.ccnv"
        )
        with self.assertRaises(SchemaVersionError):
            validate_sample_table(pq.read_table(v1_sample))

    def test_all_synthetic_germline_predict_samples_match_truth(self) -> None:
        truth = pf.load_testdata_truth("predict/germline/synthetic/truth.json")
        samples = truth["samples"]
        if not isinstance(samples, dict):
            self.fail("expected samples dict in germline truth.json")
        ref_path = pf.ensure_testdata_file(
            "reference/synthetic_cohort/germline_ref_5kb.ccnv"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in (
                "euploid_female",
                "euploid_male",
                "cnv_events_female",
                "lowpass_homdel_male",
                "sex_cnv_amp_male",
            ):
                exp = samples[name]
                if not isinstance(exp, dict):
                    self.fail(f"expected dict for {name}")
                sample_path = pf.ensure_testdata_file(
                    f"predict/germline/synthetic/{name}.ccnv"
                )
                with self.subTest(sample=name):
                    out = run_predict(
                        PredictConfig(
                            sample=sample_path,
                            reference=ref_path,
                            prefix=root / name,
                            mode=AnalysisMode.germline,
                            plot=False,
                            vcf=False,
                            threads=1,
                        )
                    )
                    if out.qc_json_path is None:
                        self.fail("expected qc_json_path")
                    qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                    self.assertEqual(qc["sex"], exp["sex"])
                    self.assertEqual(qc["cnv_summary"]["gain_calls"], exp["gain_calls"])
                    self.assertEqual(qc["cnv_summary"]["loss_calls"], exp["loss_calls"])
                    called = [
                        (s.contig, s.call, s.copy_number)
                        for s in out.segments
                        if s.call is not None
                    ]
                    expected_calls = [
                        (c["contig"], c["call"], c["copy_number"]) for c in exp["calls"]
                    ]
                    self.assertEqual(called, expected_calls)

    def test_all_synthetic_prenatal_predict_samples_match_truth(self) -> None:
        truth = pf.load_testdata_truth("predict/prenatal/synthetic/truth.json")
        samples = truth["samples"]
        if not isinstance(samples, dict):
            self.fail("expected samples dict in prenatal truth.json")
        ref_path = pf.ensure_testdata_file(
            "predict/prenatal/synthetic/prenatal_ref_synthetic.ccnv"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, is_twin in (
                ("euploid_xx", False),
                ("euploid_xy", False),
                ("trisomy_chr2_xy", False),
                ("maternal_dup_xx", False),
                ("twin_fm", True),
                ("sca_xo", False),
                ("chry_del_high_mw_xy", False),
            ):
                exp = samples[name]
                if not isinstance(exp, dict):
                    self.fail(f"expected dict for {name}")
                sample_path = pf.ensure_testdata_file(
                    f"predict/prenatal/synthetic/{name}.ccnv"
                )
                with self.subTest(sample=name):
                    out = run_predict(
                        PredictConfig(
                            sample=sample_path,
                            reference=ref_path,
                            prefix=root / name,
                            mode=AnalysisMode.prenatal,
                            twins=is_twin,
                            plot=False,
                            vcf=False,
                            threads=1,
                        )
                    )
                    if out.qc_json_path is None:
                        self.fail("expected qc_json_path")
                    qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                    pr = qc["prenatal"]
                    self.assertEqual(pr["fetal_sex"], exp["fetal_sex"])
                    self.assertEqual(pr["sca"]["call"], exp["sca_call"])
                    self.assertEqual(pr["gestation"], exp["gestation"])
                    self.assertEqual(qc["cnv_summary"]["gain_calls"], exp["gain_calls"])
                    self.assertEqual(qc["cnv_summary"]["loss_calls"], exp["loss_calls"])
                    if is_twin:
                        self.assertEqual(pr["twin"]["call"], exp["twin_call"])
                    if "frag_warnings" in exp:
                        self.assertEqual(
                            qc["fragmentomics"]["warnings"], exp["frag_warnings"]
                        )


@unittest.skipUnless(
    _RUN_REAL_TESTDATA,
    "set CCNV_RUN_REAL_TESTDATA=1 to run full real-world regression suite (~214 MB)",
)
class TestRealWorldTestdataRegression(unittest.TestCase):
    """Full regression checks against all real-world datasets in copynumber-cnv-testdata."""

    def test_real_references_validate_and_pass_ref_qc_execution(self) -> None:
        for rel, exp_count, exp_flat in (
            ("reference/schema_2.0/NIPS.GRCh38-noalt.100kb.ccnv", 1060, False),
            ("reference/schema_2.0/WGS-full-pass.GRCh38-noalt.10kb.ccnv", 79, False),
            ("reference/schema_2.0/WGS-low-pass.GRCh38-noalt.15kb.ccnv", 715, False),
            ("reference/schema_2.0/flat_equCab-T2T.100kb.ccnv", 0, True),
        ):
            with self.subTest(reference=rel):
                ref_path = pf.ensure_testdata_file(rel)
                tbl, meta = read_reference_parquet(ref_path, check_rows=True)
                self.assertEqual(meta.sample_count, exp_count)
                self.assertEqual(meta.flat, exp_flat)
                qc_out = run_reference_qc_on_table(
                    tbl, write_json=False, emit_stderr=False
                )
                self.assertIn(qc_out.overall_verdict, ("PASS", "WARN", "FAIL"))

    def test_real_germline_samples_predict_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, sample_rel, ref_rel in (
                (
                    "equcab_flat",
                    "extract/schema_2.0/sample3_equCab-T2T.ccnv",
                    "reference/schema_2.0/flat_equCab-T2T.100kb.ccnv",
                ),
                (
                    "grch38_wgs",
                    "extract/schema_2.0/sample4_GRCh38-noalt-WGS.ccnv",
                    "reference/schema_2.0/WGS-full-pass.GRCh38-noalt.10kb.ccnv",
                ),
            ):
                with self.subTest(sample=name):
                    out = run_predict(
                        PredictConfig(
                            sample=pf.ensure_testdata_file(sample_rel),
                            reference=pf.ensure_testdata_file(ref_rel),
                            prefix=root / name,
                            mode=AnalysisMode.germline,
                            plot=False,
                            vcf=False,
                            threads=2,
                        )
                    )
                    if out.qc_json_path is None:
                        self.fail("expected qc_json_path")
                    qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                    self.assertEqual(qc["mode"], "germline")
                    self.assertGreater(qc["read_metrics"]["usable_bins"], 0)

    def test_real_prenatal_twins_and_scas_match_clinical_truth(self) -> None:
        nips_ref = pf.ensure_testdata_file(
            "reference/schema_2.0/NIPS.GRCh38-noalt.100kb.ccnv"
        )
        twins_truth = pf.load_testdata_truth("predict/prenatal/twins/truth.json")
        sca_truth = pf.load_testdata_truth("predict/prenatal/sca/truth.json")
        twins_samples = twins_truth["samples"]
        sca_samples = sca_truth["samples"]
        if not isinstance(twins_samples, dict) or not isinstance(sca_samples, dict):
            self.fail("expected samples dicts in twins/sca truth.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, exp in twins_samples.items():
                if not isinstance(exp, dict):
                    self.fail(f"expected dict for twin {name}")
                with self.subTest(twin=name):
                    out = run_predict(
                        PredictConfig(
                            sample=pf.ensure_testdata_file(
                                f"predict/prenatal/twins/{exp['file']}"
                            ),
                            reference=nips_ref,
                            prefix=root / name,
                            mode=AnalysisMode.prenatal,
                            twins=True,
                            plot=False,
                            vcf=False,
                            threads=2,
                        )
                    )
                    if out.qc_json_path is None:
                        self.fail("expected qc_json_path")
                    qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                    pr = qc["prenatal"]
                    self.assertEqual(pr["gestation"], "twin")
                    self.assertEqual(pr["twin"]["call"], exp["expected_twin_call"])
                    self.assertEqual(pr["fetal_sex"], exp["expected_fetal_sex"])

            for name, exp in sca_samples.items():
                if not isinstance(exp, dict):
                    self.fail(f"expected dict for sca {name}")
                with self.subTest(sca=name):
                    out = run_predict(
                        PredictConfig(
                            sample=pf.ensure_testdata_file(
                                f"predict/prenatal/sca/{exp['file']}"
                            ),
                            reference=nips_ref,
                            prefix=root / name,
                            mode=AnalysisMode.prenatal,
                            twins=False,
                            plot=False,
                            vcf=False,
                            threads=2,
                        )
                    )
                    if out.qc_json_path is None:
                        self.fail("expected qc_json_path")
                    qc = json.loads(out.qc_json_path.read_text(encoding="utf-8"))
                    pr = qc["prenatal"]
                    self.assertEqual(pr["sca"]["call"], exp["expected_sca_call"])
                    self.assertEqual(pr["fetal_sex"], exp["expected_fetal_sex"])


if __name__ == "__main__":
    unittest.main()
