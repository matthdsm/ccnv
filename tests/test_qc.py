"""Unit tests for ccnv.qc subpackage."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import make_reference_table, make_sample_table, metadata_for

from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    NUM_FRAG_BANDS,
    NUM_PROTOTYPES_PER_BIN,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    ArtifactKind,
    BinResult,
    BranchFitArrays,
    FetalFractionResult,
    PrenatalLoDMetrics,
    ReadFilters,
    ReferenceMetadata,
    ReferenceQCConfig,
    SCAResult,
    SegmentResult,
    TwinResult,
    assemble_reference_table,
)
from ccnv.qc.predict import build_predict_qc_payload, write_predict_qc_json
from ccnv.qc.reference import (
    CHRY_MEAN_DIST_FAIL,
    CHRY_MIN_USABLE_PCT_FAIL,
    CHRY_MIN_USABLE_PCT_WARN,
    ReferenceQCError,
    _evaluate_base_support,
    compute_branch_qc,
    compute_signal_contig_metrics,
    evaluate_autosomal_or_female_verdict,
    evaluate_male_verdict,
    run_reference_qc_on_table,
    write_reference_qc_json,
)
from ccnv.qc.sample import QC_COUNTERS, build_sample_qc, empty_qc
from ccnv.ref_qc import run_reference_qc


def _make_predict_sample_table(
    contig_lengths: dict[str, int] | None = None,
    counts: list[int] | None = None,
    qc: dict[str, int] | None = None,
) -> pa.Table:
    lengths = contig_lengths or {"chr1": 200, "chr2": 100, "chrX": 100}
    vals = [250, 250, 300, 50] if counts is None else counts
    default_qc = dict(
        zip(
            (
                "effective_reads_examined",
                "effective_reads_kept",
                "autosomal_reads_kept",
                "sex_reads_kept",
                "filter_unmapped",
                "filter_mapq",
                "filter_duplicate",
                "filter_secondary",
                "filter_supplementary",
                "filter_improper_pair",
                "filter_no_start",
            ),
            (1000, 850, 800, 50, 50, 40, 30, 10, 10, 5, 5),
            strict=True,
        )
    )
    n = len(vals)
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=lengths,
        values={
            "bin_size": 100,
            "contig_order": list(lengths.keys()),
            "source_identity": {"path": "/data/test_sample.bam", "format": "bam"},
            "reference_identity": {"id": "genome"},
            "qc": default_qc if qc is None else qc,
            "filters": asdict(ReadFilters()),
        },
    )
    z32 = pa.array([0] * n, type=pa.uint32())
    cols = {
        "count": pa.array(vals, type=pa.uint32()),
        "paired_count": z32,
        "frag_len_sum": pa.array([0] * n, type=pa.uint64()),
        "short_frag_count": z32,
        "endpoint_count": z32,
        "frag_gc_sum": pa.array([0.0] * n, type=pa.float32()),
        "frag_bands": pa.nulls(n, type=pa.list_(pa.uint32(), NUM_FRAG_BANDS)),
    }
    return pa.table(cols, schema=SAMPLE_SCHEMA.with_metadata(metadata))


def _ref_model(**overrides: object) -> dict[str, object]:
    none_keys = ("sex_signal_contig", "sex_cutoff", "sample_sexes", "sex_inference")
    base: dict[str, object] = {k: None for k in none_keys} | {
        "bin_size": 100,
        "branches": ["autosomal"],
        "autosomal_contigs": ["chr1"],
        "sex_contigs": [],
        "sex_source": "none",
        "pca_components": 2,
        "nearest_bins": 5,
        "max_batch_memory_mb": 64,
        "flat": False,
    }
    return base | overrides


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

    def test_sample_qc_zero_preservation(self) -> None:
        raw = {"reads_seen": 25, "reads_kept": 20}
        raw.update(effective_reads_examined=0, effective_reads_kept=0)
        qc_dict = build_sample_qc(raw)
        eff = (qc_dict["effective_reads_examined"], qc_dict["effective_reads_kept"])
        self.assertEqual(eff, (0, 0))
        tbl = _make_predict_sample_table({"chr1": 100}, [5], qc_dict)
        rm = build_predict_qc_payload("germline", tbl, ())["read_metrics"]
        if not isinstance(rm, dict):
            self.fail("Expected read_metrics dict")
        self.assertEqual(
            (rm["effective_reads_examined"], rm["effective_reads_kept"]), (0, 0)
        )


class TestQCReference(unittest.TestCase):
    def test_compute_signal_contig_metrics(self) -> None:
        contigs = np.array(["chr1", "chrY", "chrY"])
        mean_d = np.array([0.5, 1.2, 1.4], dtype=np.float64)
        n_refs = np.array([200, 200, 200], dtype=np.int64)
        metrics = compute_signal_contig_metrics(
            contigs, "chrY", mean_d, n_refs, cutoff_outlier=2.0, min_ref_bins=100
        )
        self.assertIsNotNone(metrics)
        if metrics is None:
            self.fail("expected non-None signal contig metrics")
        self.assertEqual(metrics["signal_contig"], "chrY")
        self.assertEqual(metrics["n_bins"], 2)
        self.assertEqual(metrics["n_valid"], 2)
        self.assertAlmostEqual(float(metrics["mean_of_means"]), 1.3)

    def test_compute_signal_contig_metrics_missing(self) -> None:
        res = compute_signal_contig_metrics(
            ["chr1", "chr2"],
            "chrY",
            np.array([1.0, 2.0]),
            np.array([200, 200]),
            5.0,
            150,
        )
        self.assertIsNone(res)

    def test_compute_signal_contig_metrics_present(self) -> None:
        res = compute_signal_contig_metrics(
            ["chr1", "chrY", "chrY"],
            "chrY",
            np.array([1.0, 2.0, 6.0], dtype=np.float64),
            np.array([200, 100, 160], dtype=np.int64),
            5.0,
            150,
        )
        self.assertIsNotNone(res)
        if res is None:
            self.fail("expected non-None signal contig metrics")
        self.assertEqual(res["signal_contig"], "chrY")
        self.assertEqual(res["n_bins"], 2)
        self.assertEqual(res["n_valid"], 2)
        self.assertAlmostEqual(float(res["mean_of_means"]), 4.0)
        self.assertEqual(res["n_mean_outlier"], 1)
        self.assertEqual(res["n_low_refs"], 1)
        self.assertEqual(res["n_usable"], 1)
        self.assertAlmostEqual(float(res["usable_pct"]), 50.0)

    def test_evaluate_autosomal_verdict(self) -> None:
        def _m(
            n_valid: int, std_m: float, outlier_pct: float = 0.5, low_refs: int = 0
        ) -> dict[str, int | float]:
            return {
                "n_valid": n_valid,
                "n_low_refs": low_refs,
                "outlier_pct": outlier_pct,
                "std_of_means": std_m,
            }

        self.assertEqual(
            evaluate_autosomal_or_female_verdict(_m(100, 0.8), 150)[0], "PASS"
        )
        self.assertEqual(
            evaluate_autosomal_or_female_verdict(_m(100, 2.5, 2.0, 5), 150)[0], "WARN"
        )
        self.assertEqual(
            evaluate_autosomal_or_female_verdict(_m(100, 15.0, 0.0), 150)[0], "FAIL"
        )

    def test_evaluate_autosomal_verdict_pass(self) -> None:
        m = {"n_valid": 1000, "std_of_means": 1.2, "outlier_pct": 0.5, "n_low_refs": 0}
        verdict, msg = evaluate_autosomal_or_female_verdict(m, min_ref_bins=150)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(msg, "")

    def test_evaluate_autosomal_verdict_warn_and_fail(self) -> None:
        def _m(
            std_m: float, out_p: float = 0.5, low_r: int = 0
        ) -> dict[str, int | float]:
            return {
                "n_valid": 1000,
                "std_of_means": std_m,
                "outlier_pct": out_p,
                "n_low_refs": low_r,
            }

        verdict, msg = evaluate_autosomal_or_female_verdict(_m(2.5), min_ref_bins=150)
        self.assertEqual(verdict, "WARN")
        self.assertIn("std(per-bin mean dist)", msg)
        verdict, msg = evaluate_autosomal_or_female_verdict(_m(11.0), min_ref_bins=150)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("(high)", msg)
        verdict, msg = evaluate_autosomal_or_female_verdict(_m(1.0, 0.0, 6), 150)
        self.assertEqual(verdict, "WARN")
        self.assertIn("n_refs<150", msg)

    def test_evaluate_male_verdict(self) -> None:
        def _male(usable_pct: float) -> dict[str, object]:
            sig = {"n_valid": 10, "usable_pct": usable_pct, "mean_of_means": 2.0}
            return {
                "n_valid": 100,
                "n_low_refs": 0,
                "mean_of_means": 1.0,
                "outlier_pct": 0.5,
                "signal_contig_metrics": sig,
            }

        self.assertEqual(
            evaluate_male_verdict(_male(90.0), min_ref_bins=150)[0], "PASS"
        )
        self.assertEqual(
            evaluate_male_verdict(_male(10.0), min_ref_bins=150)[0], "FAIL"
        )

    def test_evaluate_male_verdict_heuristics(self) -> None:
        def _male(usable_pct: float, mean_m: float = 2.0) -> dict[str, object]:
            sig = {
                "signal_contig": "chrY",
                "n_valid": 100,
                "usable_pct": usable_pct,
                "mean_of_means": mean_m,
            }
            return {
                "n_valid": 1000,
                "mean_of_means": 1.0,
                "outlier_pct": 0.0,
                "n_low_refs": 0,
                "signal_contig_metrics": sig,
            }

        self.assertEqual(
            evaluate_male_verdict(_male(80.0), min_ref_bins=150)[0], "PASS"
        )

        verdict, msg = evaluate_male_verdict(
            _male(CHRY_MIN_USABLE_PCT_WARN - 5.0), min_ref_bins=150
        )
        self.assertEqual(verdict, "WARN")
        self.assertIn("usable chrY bins", msg)

        verdict, msg = evaluate_male_verdict(
            _male(CHRY_MIN_USABLE_PCT_FAIL - 5.0), min_ref_bins=150
        )
        self.assertEqual(verdict, "FAIL")
        self.assertIn("usable chrY bins", msg)

        verdict, msg = evaluate_male_verdict(
            _male(90.0, CHRY_MEAN_DIST_FAIL + 10.0), min_ref_bins=150
        )
        self.assertEqual(verdict, "FAIL")
        self.assertIn("very poor chrY", msg)

    def test_write_reference_qc_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sub" / "ref_qc.json"
            write_reference_qc_json(json_path, {"verdict": "PASS", "score": 0.99})
            self.assertTrue(json_path.is_file())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["verdict"], "PASS")

    def test_run_reference_qc_on_table(self) -> None:
        n_bins = 200
        st = np.arange(n_bins) * 100
        z32, o32 = np.zeros(n_bins, dtype=np.float32), np.ones(n_bins, dtype=np.float32)
        mdl = {"mode": "germline", "bin_size": 100, "nearest_bins": 300, "flat": False}
        table = make_reference_table(
            branches=[BRANCH_AUTOSOMAL] * n_bins,
            contigs=["chr1"] * n_bins,
            starts=st,
            ends=st + 100,
            contig_lengths={"chr1": 20000},
            pca_mean=z32,
            mean_distance=o32,
            null_mean=z32,
            bio_dispersion=o32,
            sample_identities=[{"path": "s1"}],
            model=mdl,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_reference_qc_on_table(
                table, Path(tmpdir) / "test_ref", write_json=True, emit_stderr=False
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertTrue(outputs.json_path.is_file())

    def test_run_reference_qc_on_synthetic_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            n_bins = 20
            st = [i * 100 for i in range(n_bins)]
            table = make_reference_table(
                branches=["autosomal"] * n_bins,
                contigs=["chr1"] * n_bins,
                starts=st,
                ends=[s + 100 for s in st],
                contig_lengths={"chr1": 2000},
                gc=[0.42] * n_bins,
                mappability=[0.95] * n_bins,
                pca_mean=[1.0] * n_bins,
                pca_loadings=[[0.1, 0.2]] * n_bins,
                mean_distance=[0.55] * n_bins,
                null_mean=[0.0] * n_bins,
                null_std=[0.1] * n_bins,
                sample_identities=[{"path": "s1.ccnv"}, {"path": "s2.ccnv"}],
                model=_ref_model(),
            )
            outputs = run_reference_qc_on_table(
                table, prefix=Path(tmpdir) / "ref", write_json=True, emit_stderr=False
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertEqual(outputs.worst_severity, 0)
            self.assertTrue(outputs.json_path.is_file())
            saved = json.loads(outputs.json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["overall_verdict"], "PASS")
            self.assertIn("autosomal", saved["branches"])
            self.assertEqual(saved["cohort"]["sample_count"], 2)

    def test_run_reference_qc_flat_fasta_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            n_bins = 10
            st = [i * 100 for i in range(n_bins)]
            table = make_reference_table(
                branches=["autosomal"] * n_bins,
                contigs=["chr1"] * n_bins,
                starts=st,
                ends=[s + 100 for s in st],
                contig_lengths={"chr1": 1000},
                gc=[0.42] * n_bins,
                mappability=[0.95] * n_bins,
                pca_mean=[1.0] * n_bins,
                pca_loadings=[[0.0, 0.0]] * n_bins,
                mean_distance=[0.15] * n_bins,
                null_mean=[0.0] * n_bins,
                null_std=[0.1] * n_bins,
                sample_identities=[],
                model=_ref_model(flat=True),
            )
            outputs = run_reference_qc_on_table(
                table, Path(tmpdir) / "flat_ref", write_json=True, emit_stderr=False
            )
            self.assertEqual(outputs.overall_verdict, "WARN")
            self.assertEqual(outputs.worst_severity, 1)
            messages = outputs.data["messages"]
            if not isinstance(messages, list):
                self.fail("expected messages list")
            self.assertTrue(any("flat FASTA" in str(msg) for msg in messages))

    def test_run_reference_qc_file_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist.ccnv"
            with self.assertRaises(ReferenceQCError):
                run_reference_qc(ReferenceQCConfig(reference=nonexistent))

    def test_evaluate_reference_qc_prenatal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
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
            model = _ref_model(
                bin_size=5000,
                prenatal_bin_size=100000,
                mode="prenatal",
                branches=["autosomal", "prenatal_autosomal", "prenatal_ffy"],
                sex_contigs=["chrY"],
                sex_source="prenatal_model",
                sample_sexes=["female", "male"],
            )
            prenatal_meta = {
                "prenatal_bin_size": 100000,
                "yfrac_threshold": 0.005,
                "ffy_slope": 1.0,
                "olm_weights": [0.1],
                "olm_bias": 0.0,
                "female_chrx_ratio_mean": 1.0,
                "female_chrx_ratio_std": 0.015,
                "alpha_1": -0.5,
            }
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
                model=model,
                extra_metadata={"prenatal": prenatal_meta},
            )
            outputs = run_reference_qc_on_table(
                table,
                Path(tmpdir) / "prenatal_qc_ref",
                write_json=True,
                emit_stderr=False,
            )
            self.assertEqual(outputs.overall_verdict, "PASS")
            self.assertIn("prenatal", outputs.data)
            p_qc = outputs.data["prenatal"]
            if not isinstance(p_qc, dict):
                self.fail("expected prenatal dict")
            self.assertEqual(p_qc["usable_autosomal_bins"], 1)
            self.assertEqual(p_qc["informative_ffy_bins"], 15)
            self.assertAlmostEqual(float(p_qc["yfrac_threshold"]), 0.005)
            self.assertAlmostEqual(float(p_qc["female_chrx_ratio_std"]), 0.015)

            branches_dict = outputs.data["branches"]
            if not isinstance(branches_dict, dict):
                self.fail("expected branches dict")
            self.assertIn("autosomal", branches_dict)
            self.assertNotIn("prenatal_autosomal", branches_dict)
            self.assertNotIn("prenatal_ffy", branches_dict)

    def test_ref_qc_fails_male_branch_when_100pct_chry_bins_masked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            n_auto, n_chry = 20, 10
            total = n_auto + n_chry
            starts = [i * 100 for i in range(n_auto)] + [i * 100 for i in range(n_chry)]
            ends = [s + 100 for s in starts]
            model = _ref_model(
                branches=[BRANCH_MALE_SEX],
                sex_contigs=["chrY"],
                sex_signal_contig="chrY",
                sex_cutoff=0.05,
                sex_source="inferred",
            )
            table = make_reference_table(
                branches=[BRANCH_MALE_SEX] * total,
                contigs=["chr1"] * n_auto + ["chrY"] * n_chry,
                starts=starts,
                ends=ends,
                contig_lengths={"chr1": n_auto * 100, "chrY": n_chry * 100},
                masked=[False] * n_auto + [True] * n_chry,
                gc=[0.42] * total,
                mappability=[0.95] * total,
                pca_mean=[1.0] * total,
                pca_loadings=[[0.1, 0.2]] * total,
                mean_distance=[0.55] * total,
                null_mean=[0.0] * total,
                bio_dispersion=[0.05] * total,
                model=model,
                sample_identities=[{"path": "s1.ccnv"}, {"path": "s2.ccnv"}],
            )
            verdict = run_reference_qc_on_table(
                table, prefix=Path(tmpdir) / "ref", write_json=False, emit_stderr=False
            )
            self.assertEqual(verdict.branches[BRANCH_MALE_SEX].status, "FAIL")

    def test_compute_branch_qc_male_chry_usable_pct_uses_total_branch_chry_bins(
        self,
    ) -> None:
        n_x, n_y = 20, 100
        starts = np.concatenate(
            [
                np.arange(n_x, dtype=np.int64) * 5000,
                np.arange(n_y, dtype=np.int64) * 5000,
            ]
        )
        masked = np.zeros(n_x + n_y, dtype=np.bool_)
        masked[n_x + 1 :] = True
        pids = np.zeros((n_x + n_y, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pweights = np.zeros((n_x + n_y, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        pweights[~masked, 0] = np.float16(1.0)
        pids[: n_x + 1, 0] = np.uint16(1)

        fit = BranchFitArrays(
            branch=BRANCH_MALE_SEX,
            contigs=("chrX",) * n_x + ("chrY",) * n_y,
            starts=starts,
            ends=starts + 5000,
            masked=masked,
            pca_mean=np.ones(n_x + n_y, dtype=np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.full(n_x + n_y, 0.5, dtype=np.float32),
        )
        ref_meta = ReferenceMetadata(
            bin_size=5000,
            contig_lengths={"chrX": n_x * 5000, "chrY": n_y * 5000},
            branches=(BRANCH_MALE_SEX,),
            autosomal_contigs=(),
            sex_contigs=("chrX", "chrY"),
            sex_signal_contig="chrY",
            pca_components=1,
        )
        table = assemble_reference_table(
            [fit],
            contig_order=("chrX", "chrY"),
            metadata=ref_meta.build_arrow_metadata(),
        )
        qc = compute_branch_qc(table, BRANCH_MALE_SEX, "chrY", min_ref_bins=1)
        if qc is None:
            self.fail("expected non-None male_sex branch QC dictionary")
        self.assertEqual(qc["verdict"], "FAIL")
        sig = qc["signal_contig_metrics"]
        if not isinstance(sig, dict):
            self.fail("expected signal_contig_metrics dictionary")
        self.assertEqual(sig["n_bins"], 100)
        self.assertAlmostEqual(float(sig["usable_pct"]), 1.0)

    def test_reference_qc_low_refs_uses_total_unmasked_bins(self) -> None:
        n_val, (verdict, msg) = _evaluate_base_support(
            {"n_bins": 1000, "n_masked": 0, "n_valid": 200, "n_low_refs": 4},
            min_ref_bins=3,
        )
        self.assertEqual(n_val, 200)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(msg, "")

    def test_ref_qc_scales_distance_thresholds_per_branch_pca_rank(self) -> None:
        rng = np.random.default_rng(11)
        n_bins = 20
        starts = np.tile(np.arange(10, dtype=np.int64) * 5000, 2)
        pids = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pids[:10, 0] = 1
        pweights = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (n_bins, 1)
        )
        fit = BranchFitArrays(
            branch=BRANCH_AUTOSOMAL,
            contigs=("chr1",) * 10 + ("chr2",) * 10,
            starts=starts,
            ends=starts + 5000,
            masked=np.zeros(n_bins, dtype=np.bool_),
            pca_mean=rng.uniform(80.0, 120.0, size=n_bins).astype(np.float32),
            pca_loadings=rng.normal(size=(n_bins, 1)).astype(np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=np.linspace(1.0, 18.0, n_bins, dtype=np.float32),
            null_mean=np.zeros(n_bins, dtype=np.float32),
            bio_dispersion=np.full(n_bins, 0.01, dtype=np.float32),
        )
        ref_meta = ReferenceMetadata(
            bin_size=5000,
            contig_lengths={"chr1": 50_000, "chr2": 50_000},
            branches=(BRANCH_AUTOSOMAL,),
            autosomal_contigs=("chr1", "chr2"),
            sex_contigs=(),
            pca_components=49,
            branch_pca_ranks={BRANCH_AUTOSOMAL: 1},
        )
        ref_tbl = assemble_reference_table(
            [fit],
            contig_order=("chr1", "chr2"),
            metadata=ref_meta.build_arrow_metadata(),
        )
        qc_out = run_reference_qc_on_table(ref_tbl, write_json=False, emit_stderr=False)
        self.assertEqual(qc_out.overall_verdict, "WARN")


class TestQCPredict(unittest.TestCase):
    def test_build_predict_qc_payload_germline(self) -> None:
        simple_table = make_sample_table([10, 20, 30], bin_size=100)
        simple_segments = (
            SegmentResult("chr1", 0, 100, 0.5, 3.2, 1.0, 1, "gain"),
            SegmentResult("chr1", 100, 300, -0.5, -3.2, 2.0, 2, "loss"),
        )
        simple_payload = build_predict_qc_payload(
            "germline", simple_table, simple_segments
        )
        self.assertEqual(simple_payload["mode"], "germline")
        simple_cnv = simple_payload["cnv_summary"]
        if not isinstance(simple_cnv, dict):
            self.fail("expected cnv_summary dict")
        self.assertEqual(simple_cnv["total_segments"], 2)
        self.assertEqual(simple_cnv["gain_calls"], 1)
        self.assertEqual(simple_cnv["loss_calls"], 1)
        self.assertIsNone(simple_payload["prenatal"])

        table = _make_predict_sample_table()
        segments = (
            SegmentResult("chr1", 0, 100, 0.4, 3.5, 1.0, 1, "gain"),
            SegmentResult("chr1", 100, 200, -0.5, -3.2, 1.0, 1, "loss"),
            SegmentResult("chr2", 0, 100, 0.0, 0.1, 1.0, 1, None),
        )
        payload = build_predict_qc_payload("germline", table, segments)

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["mode"], "germline")
        self.assertEqual(payload["sample"], "test_sample")
        self.assertIsNone(payload["sex"])
        self.assertIsNone(payload["prenatal"])

        read_metrics = payload["read_metrics"]
        if not isinstance(read_metrics, dict):
            self.fail("expected read_metrics dict")
        self.assertEqual(read_metrics["effective_reads_examined"], 1000)
        self.assertEqual(read_metrics["effective_reads_kept"], 850)
        self.assertEqual(read_metrics["autosomal_reads_kept"], 800)
        self.assertEqual(read_metrics["sex_reads_kept"], 50)
        self.assertEqual(read_metrics["total_bins"], 4)
        self.assertEqual(read_metrics["usable_bins"], 4)
        self.assertEqual(read_metrics["masked_bins"], 0)
        self.assertAlmostEqual(float(read_metrics["mean_bin_count"]), 212.5)

        filter_drops = read_metrics["filter_drops"]
        if not isinstance(filter_drops, dict):
            self.fail("expected filter_drops dict")
        self.assertEqual(filter_drops["unmapped"], 50)
        self.assertEqual(filter_drops["duplicate"], 30)

        cnv_summary = payload["cnv_summary"]
        if not isinstance(cnv_summary, dict):
            self.fail("expected cnv_summary dict")
        self.assertEqual(cnv_summary["total_segments"], 3)
        self.assertEqual(cnv_summary["gain_calls"], 1)
        self.assertEqual(cnv_summary["loss_calls"], 1)
        self.assertEqual(cnv_summary["discordant_segments"], 0)

    def test_build_predict_qc_payload_prenatal(self) -> None:
        table = _make_predict_sample_table()
        sca = SCAResult("XY", False, 0.98, -0.2, ff_delta=0.005)
        lod = PrenatalLoDMetrics(lod_ff_35mb=0.015, lod_ff_3mb=0.048)
        ff_args = ("M", 0.12, "FFY", 0.12, 0.115, 0.118, 0.94, 0.006, 0.004, sca)
        prenatal_result = FetalFractionResult(
            *ff_args,
            low_ff_warning=False,
            warnings=(),
            ff_size=0.122,
            active_ff_se=0.008,
            lod_metrics=lod,
            effective_ff=0.14,
        )
        s1_pos = ("chr1", 0, 100, 0.08, 3.1, 1.0, 1, "gain")
        s2_pos = ("chr2", 0, 100, 0.01, 3.1, 1.0, 1, "gain")
        segments = (
            SegmentResult(
                *s1_pos,
                expected_ratio=0.084,
                implied_ff=0.115,
                discordant=False,
                origin_call="FETAL_CONSTITUTIONAL",
            ),
            SegmentResult(
                *s2_pos,
                expected_ratio=0.084,
                implied_ff=0.014,
                discordant=True,
                discordance_reason="SUB_FETAL_RATIO",
                origin_call="MATERNAL_UNINHERITED",
            ),
        )
        payload = build_predict_qc_payload(
            "prenatal", table, segments, prenatal_result=prenatal_result
        )
        self.assertEqual(payload["mode"], "prenatal")
        self.assertEqual(payload["sex"], "male")
        cnv_summary = payload["cnv_summary"]
        if not isinstance(cnv_summary, dict):
            self.fail("expected cnv_summary dict")
        self.assertEqual(cnv_summary["discordant_segments"], 1)

        prenatal = payload["prenatal"]
        if not isinstance(prenatal, dict):
            self.fail("expected prenatal dict")
        self.assertEqual(prenatal["fetal_sex"], "M")
        self.assertEqual(prenatal["active_ff"], 0.12)
        self.assertEqual(prenatal["effective_ff"], 0.14)
        self.assertEqual(prenatal["active_ff_se"], 0.008)
        self.assertEqual(prenatal["active_ff_source"], "FFY")
        self.assertEqual(prenatal["ffy"], 0.12)
        self.assertEqual(prenatal["ffx"], 0.115)
        self.assertEqual(prenatal["ffa"], 0.118)
        self.assertEqual(prenatal["ff_size"], 0.122)
        self.assertEqual(prenatal["lod_ff_35mb"], 0.015)
        self.assertEqual(prenatal["lod_ff_3mb"], 0.048)
        self.assertFalse(prenatal["low_ff_warning"])
        self.assertEqual(
            prenatal["origin_summary"],
            {"FETAL_CONSTITUTIONAL": 1, "MATERNAL_UNINHERITED": 1},
        )

        sca_dict = prenatal["sca"]
        if not isinstance(sca_dict, dict):
            self.fail("expected sca dict")
        self.assertEqual(sca_dict["call"], "XY")
        self.assertFalse(sca_dict["is_aneuploid"])
        self.assertEqual(sca_dict["confidence"], 0.98)
        self.assertIsNone(prenatal["twin"])
        self.assertEqual(prenatal["gestation"], "singleton")
        self.assertIsNone(prenatal["fetal_sexes"])
        self.assertIsNone(prenatal["male_ff"])
        self.assertIsNone(prenatal["female_ff"])
        self.assertIsNone(prenatal["twin_confidence"])

    def test_build_predict_qc_payload_prenatal_twin(self) -> None:
        table = _make_predict_sample_table()
        sca = SCAResult("XY", False, 0.9912, -6.53, ff_delta=0.011)
        twin = TwinResult(
            call="FM",
            fetal_sexes=("F", "M"),
            is_twin=True,
            twin_probability=0.92,
            confidence=0.9718,
            posteriors={"FF": 0.0001, "FM": 0.9718, "MM": 0.0281},
            male_ff=0.143,
            female_ff=0.075,
        )
        prenatal_result = FetalFractionResult(
            "M",
            0.218,
            "FF_SIZE",
            0.143,
            0.15,
            0.101,
            0.81,
            0.00023,
            0.000043,
            sca,
            twin=twin,
            gestation="twin",
            fetal_sexes=("F", "M"),
            male_ff=0.143,
            female_ff=0.075,
            twin_confidence=0.9718,
        )
        payload = build_predict_qc_payload(
            "prenatal", table, (), prenatal_result=prenatal_result
        )
        prenatal = payload["prenatal"]
        if not isinstance(prenatal, dict):
            self.fail("expected prenatal dict")
        self.assertEqual(prenatal["gestation"], "twin")
        self.assertEqual(prenatal["fetal_sexes"], ["F", "M"])
        self.assertEqual(prenatal["male_ff"], 0.143)
        self.assertEqual(prenatal["female_ff"], 0.075)
        self.assertEqual(prenatal["twin_confidence"], 0.9718)
        self.assertEqual(
            prenatal["sca"],
            {
                "call": "XY",
                "is_aneuploid": False,
                "confidence": 0.9912,
                "chrx_zscore": -6.53,
                "ff_delta": 0.011,
            },
        )
        self.assertEqual(
            prenatal["twin"],
            {
                "call": "FM",
                "fetal_sexes": ["F", "M"],
                "is_twin": True,
                "twin_probability": 0.92,
                "confidence": 0.9718,
                "posteriors": {"FF": 0.0001, "FM": 0.9718, "MM": 0.0281},
                "male_ff": 0.143,
                "female_ff": 0.075,
            },
        )

    def test_build_predict_qc_payload_sex_resolution(self) -> None:
        table = _make_predict_sample_table()
        zeros = np.zeros(1, dtype=np.float64)
        ones = np.ones(1, dtype=np.float64)
        unmasked = np.zeros(1, dtype=np.bool_)
        ref_sz = np.ones(1, dtype=np.int64)

        def _bin(branch: str, contig: str) -> BinResult:
            return BinResult(
                branch,
                (contig,),
                np.array([0], dtype=np.int64),
                np.array([100], dtype=np.int64),
                zeros,
                zeros,
                ones,
                ref_sz,
                unmasked,
            )

        self.assertEqual(
            build_predict_qc_payload("germline", table, (), sex="female")["sex"],
            "female",
        )
        self.assertEqual(
            build_predict_qc_payload("germline", table, (), sex="male")["sex"],
            "male",
        )
        self.assertEqual(
            build_predict_qc_payload(
                "germline",
                table,
                (),
                bin_results=(
                    _bin(BRANCH_AUTOSOMAL, "chr1"),
                    _bin(BRANCH_FEMALE_SEX, "chrX"),
                ),
            )["sex"],
            "female",
        )
        self.assertEqual(
            build_predict_qc_payload(
                "germline",
                table,
                (),
                bin_results=(
                    _bin(BRANCH_AUTOSOMAL, "chr1"),
                    _bin(BRANCH_MALE_SEX, "chrY"),
                ),
            )["sex"],
            "male",
        )
        self.assertIsNone(
            build_predict_qc_payload(
                "germline", table, (), bin_results=(_bin(BRANCH_AUTOSOMAL, "chr1"),)
            )["sex"]
        )

        sca_xx = SCAResult("XX", False, 0.99, 0.0)
        prenatal_f = FetalFractionResult(
            "F", 0.10, "FFA", None, 0.10, 0.10, 1.0, 0.0, 0.004, sca_xx
        )
        self.assertEqual(
            build_predict_qc_payload(
                "prenatal",
                table,
                (),
                prenatal_result=prenatal_f,
                bin_results=(_bin(BRANCH_FEMALE_SEX, "chrX"),),
            )["sex"],
            "female",
        )
        self.assertIsNone(build_predict_qc_payload("prenatal", table, ())["sex"])

        bad_kwargs: dict[str, object] = {"sex": "autosomal"}
        with self.assertRaisesRegex(ValueError, "sex must be 'female' or 'male'"):
            build_predict_qc_payload("germline", table, (), **bad_kwargs)

    def test_write_predict_qc_json(self) -> None:
        table = _make_predict_sample_table()
        payload = build_predict_qc_payload("germline", table, (), sex="female")
        payload["float_val"] = np.float64(0.123456789)
        payload["int_val"] = np.int64(42)
        payload["bool_val"] = np.bool_(True)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "sub" / "predict_qc.json"
            out = write_predict_qc_json(json_path, payload)
            self.assertEqual(out, json_path)
            self.assertTrue(json_path.is_file())

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["mode"], "germline")
            self.assertEqual(loaded["sample"], "test_sample")
            self.assertEqual(loaded["sex"], "female")
            self.assertIn("read_metrics", loaded)
            self.assertIn("cnv_summary", loaded)
            self.assertEqual(loaded["float_val"], 0.123457)
            self.assertEqual(loaded["int_val"], 42)
            self.assertTrue(loaded["bool_val"])


if __name__ == "__main__":
    unittest.main()
