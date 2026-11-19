"""Functional and regression tests covering synthetic and real-world testdata surfaces."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import published_fixtures as pf
import pyarrow.parquet as pq

from ccnv.features.binning import build_coordinate_grid
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
    SegmentResult,
    validate_sample_table,
)
from ccnv.predict import run as run_predict
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.reference import run as run_reference

_RUN_REAL_TESTDATA = os.environ.get("CCNV_RUN_REAL_TESTDATA") == "1"


@dataclass(frozen=True)
class _MergedCall:
    contig: str
    start: int
    end: int
    call: str
    copy_number: int | None
    discordant: bool
    discordance_reason: str | None
    origin_call: str | None


def _merge_adjacent_calls(segments: Sequence[SegmentResult]) -> list[_MergedCall]:
    """Merge contiguous called segments sharing identical contig, call, and state."""
    called = sorted(
        (s for s in segments if s.call is not None),
        key=lambda s: (s.contig, s.start, s.end),
    )
    merged: list[_MergedCall] = []
    for seg in called:
        assert seg.call is not None
        item = _MergedCall(
            contig=seg.contig,
            start=int(seg.start),
            end=int(seg.end),
            call=seg.call,
            copy_number=seg.copy_number,
            discordant=bool(seg.discordant),
            discordance_reason=seg.discordance_reason,
            origin_call=seg.origin_call,
        )
        if (
            merged
            and merged[-1].contig == item.contig
            and merged[-1].end == item.start
            and merged[-1].call == item.call
            and merged[-1].copy_number == item.copy_number
            and merged[-1].origin_call == item.origin_call
            and merged[-1].discordant == item.discordant
            and merged[-1].discordance_reason == item.discordance_reason
        ):
            prev = merged[-1]
            merged[-1] = _MergedCall(
                contig=prev.contig,
                start=prev.start,
                end=item.end,
                call=prev.call,
                copy_number=prev.copy_number,
                discordant=prev.discordant,
                discordance_reason=prev.discordance_reason,
                origin_call=prev.origin_call,
            )
        else:
            merged.append(item)
    return merged


def _reciprocal_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    denom = max(a_end - a_start, b_end - b_start, 1)
    return float(inter / denom)


class TestSyntheticTestdataRegression(unittest.TestCase):
    """Fast deterministic tests exercising every synthetic fixture in copynumber-cnv-testdata."""

    def _assert_planted_events(
        self,
        segments: Sequence[SegmentResult],
        planted_events: Sequence[object],
        *,
        check_copy_number: bool = False,
        check_prenatal_origin: bool = False,
        min_overlap: float = 0.85,
    ) -> None:
        observed = _merge_adjacent_calls(segments)
        self.assertEqual(
            len(observed),
            len(planted_events),
            f"expected {len(planted_events)} planted events, observed {observed}",
        )
        unmatched = list(observed)
        for raw_ev in planted_events:
            if not isinstance(raw_ev, dict):
                self.fail(f"expected event dict, got {type(raw_ev).__name__}")
            e_contig = str(raw_ev["contig"])
            e_start = int(raw_ev["start"])
            e_end = int(raw_ev["end"])
            e_call = str(raw_ev["call"])
            match_idx: int | None = None
            for idx, obs in enumerate(unmatched):
                if obs.contig != e_contig or obs.call != e_call:
                    continue
                if (
                    _reciprocal_overlap(obs.start, obs.end, e_start, e_end)
                    >= min_overlap
                ):
                    match_idx = idx
                    break
            self.assertIsNotNone(
                match_idx,
                f"no matching call (>= {min_overlap:.0%} overlap) for planted {raw_ev} in {observed}",
            )
            assert match_idx is not None
            obs = unmatched.pop(match_idx)
            if check_copy_number:
                self.assertEqual(obs.copy_number, raw_ev["copy_number"])
            if check_prenatal_origin:
                self.assertEqual(obs.discordant, bool(raw_ev["discordant"]))
                self.assertEqual(obs.discordance_reason, raw_ev["discordance_reason"])
                self.assertEqual(obs.origin_call, raw_ev["origin_call"])
        self.assertEqual(unmatched, [], f"unexpected unplanted calls: {unmatched}")

    def test_synthetic_tracks_load_and_agree_between_bigwig_and_bed(self) -> None:
        truth = pf.load_testdata_truth("reference/synthetic_cohort/truth.json")
        raw_lengths = pf.require_dict(truth, "contig_lengths")
        contig_lengths = {
            str(k): int(v) for k, v in raw_lengths.items() if isinstance(v, int)
        }
        bin_size = (
            int(truth["bin_size"]) if isinstance(truth["bin_size"], int) else 5000
        )
        contigs, s_arr, e_arr, _ = build_coordinate_grid(contig_lengths, bin_size)

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

        bands = load_cytobands_track(cyto_bed, contig_lengths)
        self.assertEqual(len(bands), len(contig_lengths) * 4)

    def test_synthetic_cohort_samplesheets_rebinning_and_negative_contracts(
        self,
    ) -> None:
        ss_yaml = pf.ensure_testdata_samplesheet(
            "reference/synthetic_cohort/samplesheet_germline.yaml"
        )
        ss_json = pf.ensure_testdata_samplesheet(
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
        pe_ctrl = yaml_entries[0].filepath
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

    def test_all_synthetic_germline_predict_samples_match_planted_truth(self) -> None:
        truth = pf.load_testdata_truth("predict/germline/synthetic/truth.json")
        samples = pf.require_dict(truth, "samples")
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
                exp = pf.require_dict(samples, name)
                sample_path = pf.ensure_testdata_file(
                    f"predict/germline/synthetic/{exp['file']}"
                )
                events = exp["events"]
                if not isinstance(events, list):
                    self.fail(f"expected events list for {name}")
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
                    self._assert_planted_events(
                        out.segments, events, check_copy_number=True
                    )

    def test_all_synthetic_prenatal_predict_samples_match_planted_truth(self) -> None:
        truth = pf.load_testdata_truth("predict/prenatal/synthetic/truth.json")
        samples = pf.require_dict(truth, "samples")
        ref_path = pf.ensure_testdata_file(
            "predict/prenatal/synthetic/prenatal_ref_synthetic.ccnv"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, twin_modes in (
                ("euploid_xx", (False,)),
                ("euploid_xy", (False,)),
                ("trisomy_chr2_xy", (False,)),
                ("maternal_dup_xx", (False,)),
                ("twin_fm", (False, True)),
                ("sca_xo", (False,)),
                ("chry_del_high_mw_xy", (False,)),
            ):
                exp = pf.require_dict(samples, name)
                sample_path = pf.ensure_testdata_file(
                    f"predict/prenatal/synthetic/{exp['file']}"
                )
                events = exp["events"]
                if not isinstance(events, list):
                    self.fail(f"expected events list for {name}")
                planted_ff = pf.require_float(exp, "planted_ff")
                exp_warnings = pf.optional_str_list(exp, "frag_warnings")
                for twins_flag in twin_modes:
                    with self.subTest(sample=name, twins=twins_flag):
                        out = run_predict(
                            PredictConfig(
                                sample=sample_path,
                                reference=ref_path,
                                prefix=root / f"{name}_{twins_flag}",
                                mode=AnalysisMode.prenatal,
                                twins=twins_flag,
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
                        ff_tol = 0.04 if name != "chry_del_high_mw_xy" else 0.05
                        self.assertAlmostEqual(
                            float(pr["active_ff"]), planted_ff, delta=ff_tol
                        )
                        self.assertEqual(qc["fragmentomics"]["warnings"], exp_warnings)
                        if name == "twin_fm":
                            self.assertEqual(pr["twin"]["call"], exp["twin_call"])
                            self.assertGreater(
                                float(pr["twin"]["twin_probability"]), 0.8
                            )
                            self.assertAlmostEqual(
                                float(pr["twin"]["male_ff"]),
                                pf.require_float(exp, "planted_male_ff"),
                                delta=0.03,
                            )
                            self.assertAlmostEqual(
                                float(pr["twin"]["female_ff"]),
                                pf.require_float(exp, "planted_female_ff"),
                                delta=0.03,
                            )
                        self._assert_planted_events(
                            out.segments, events, check_prenatal_origin=True
                        )


@unittest.skipUnless(
    _RUN_REAL_TESTDATA,
    "set CCNV_RUN_REAL_TESTDATA=1 to run full real-world regression suite (~214 MB)",
)
class TestRealWorldTestdataRegression(unittest.TestCase):
    """Full regression checks against all real-world datasets in copynumber-cnv-testdata."""

    def test_real_references_validate_and_match_expected_ref_qc_verdicts(self) -> None:
        cases = (
            (
                "reference/schema_2.0/NIPS.GRCh38-noalt.100kb.ccnv",
                1060,
                False,
                "FAIL",
                "usable chrY bins",
            ),
            (
                "reference/schema_2.0/WGS-full-pass.GRCh38-noalt.10kb.ccnv",
                79,
                False,
                "FAIL",
                "usable chrY bins",
            ),
            (
                "reference/schema_2.0/WGS-low-pass.GRCh38-noalt.15kb.ccnv",
                715,
                False,
                "FAIL",
                "usable chrY bins",
            ),
            (
                "reference/schema_2.0/flat_equCab-T2T.100kb.ccnv",
                0,
                True,
                "WARN",
                "flat FASTA",
            ),
        )
        for rel, exp_count, exp_flat, exp_verdict, exp_msg in cases:
            with self.subTest(reference=rel):
                ref_path = pf.ensure_testdata_file(rel)
                tbl, meta = read_reference_parquet(ref_path, check_rows=True)
                self.assertEqual(meta.sample_count, exp_count)
                self.assertEqual(meta.flat, exp_flat)
                qc_out = run_reference_qc_on_table(
                    tbl, write_json=False, emit_stderr=False
                )
                self.assertEqual(qc_out.overall_verdict, exp_verdict)
                messages = qc_out.data.get("messages", [])
                self.assertTrue(
                    isinstance(messages, list)
                    and any(exp_msg in str(m) for m in messages),
                    f"expected {exp_msg!r} in ref-qc messages: {messages}",
                )

    def test_real_germline_samples_predict_with_calibrated_qc(self) -> None:
        cases = (
            (
                "equcab_flat",
                "extract/schema_2.0/sample3_equCab-T2T.ccnv",
                "reference/schema_2.0/flat_equCab-T2T.100kb.ccnv",
                None,
                20_000,
                3.0,
                100,
            ),
            (
                "grch38_wgs",
                "extract/schema_2.0/sample4_GRCh38-noalt-WGS.ccnv",
                "reference/schema_2.0/WGS-full-pass.GRCh38-noalt.10kb.ccnv",
                "female",
                250_000,
                1.35,
                60,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for (
                name,
                sample_rel,
                ref_rel,
                exp_sex,
                min_bins,
                max_lgc,
                max_calls,
            ) in cases:
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
                    self.assertEqual(qc["sex"], exp_sex)
                    self.assertGreaterEqual(qc["read_metrics"]["usable_bins"], min_bins)
                    lgc = float(qc["read_metrics"]["lambda_gc"])
                    self.assertTrue(math.isfinite(lgc) and 0.5 <= lgc <= max_lgc, lgc)
                    total_calls = int(qc["cnv_summary"]["gain_calls"]) + int(
                        qc["cnv_summary"]["loss_calls"]
                    )
                    self.assertLessEqual(total_calls, max_calls)

    def test_real_prenatal_twins_and_scas_match_clinical_truth(self) -> None:
        nips_ref = pf.ensure_testdata_file(
            "reference/schema_2.0/NIPS.GRCh38-noalt.100kb.ccnv"
        )
        twins_truth = pf.load_testdata_truth("predict/prenatal/twins/truth.json")
        sca_truth = pf.load_testdata_truth("predict/prenatal/sca/truth.json")
        twins_samples = pf.require_dict(twins_truth, "samples")
        sca_samples = pf.require_dict(sca_truth, "samples")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in twins_samples:
                exp = pf.require_dict(twins_samples, name)
                with self.subTest(twin=name):
                    sample_p = pf.ensure_testdata_file(
                        f"predict/prenatal/twins/{exp['file']}"
                    )
                    out = run_predict(
                        PredictConfig(
                            sample=sample_p,
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
                    act_ff = float(pr["active_ff"])
                    self.assertTrue(0.02 <= act_ff <= 0.45, act_ff)
                    if exp["expected_twin_call"] == "FM":
                        m_ff = float(pr["twin"]["male_ff"])
                        f_ff = float(pr["twin"]["female_ff"])
                        self.assertGreater(m_ff, 0.01)
                        self.assertGreater(f_ff, 0.01)
                        self.assertLess(m_ff, act_ff)

            # Verify automatic mixed-sex twin detection without --twins on twin_fm_01
            fm_auto_exp = pf.require_dict(twins_samples, "twin_fm_01")
            with self.subTest(twin_auto="twin_fm_01"):
                out_auto = run_predict(
                    PredictConfig(
                        sample=pf.ensure_testdata_file(
                            f"predict/prenatal/twins/{fm_auto_exp['file']}"
                        ),
                        reference=nips_ref,
                        prefix=root / "twin_fm_01_auto",
                        mode=AnalysisMode.prenatal,
                        twins=False,
                        plot=False,
                        vcf=False,
                        threads=2,
                    )
                )
                if out_auto.qc_json_path is None:
                    self.fail("expected qc_json_path")
                qc_auto = json.loads(out_auto.qc_json_path.read_text(encoding="utf-8"))
                self.assertEqual(qc_auto["prenatal"]["twin"]["call"], "FM")
                self.assertGreater(
                    float(qc_auto["prenatal"]["twin"]["twin_probability"]), 0.5
                )

            for name in sca_samples:
                exp = pf.require_dict(sca_samples, name)
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
                    act_ff = float(pr["active_ff"])
                    self.assertTrue(0.02 <= act_ff <= 0.45, act_ff)


if __name__ == "__main__":
    unittest.main()
