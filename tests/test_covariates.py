"""Unit and end-to-end tests for GC, mappability, and fragmentomic covariates."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pyBigWig
import pysam
from helpers import write_synthetic_sample
from typer.testing import CliRunner

import ccnv.predict
from ccnv.cli import app
from ccnv.features.binning import compute_bin_exposure
from ccnv.features.covariates import (
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.features.normalization import BranchModel, normalize_sample_branch
from ccnv.io.parquet import read_reference_parquet
from ccnv.io.tracks import load_gc_track
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    DEFAULT_MIN_MAPPABILITY,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    CovariateError,
    PredictConfig,
    ReferenceConfig,
    SamplesheetEntry,
    parquet_path,
)
from ccnv.models.schemas import decode_metadata
from ccnv.predict import build_branch_model
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.stats.regression import (
    build_exogenous_covariate_basis,
    fit_poisson_covariate_trends,
)


def _write_bigwig(
    path: Path,
    chroms: dict[str, int],
    entries: list[tuple[str, int, int, float]],
) -> Path:
    bw = pyBigWig.open(str(path), "w")
    if bw is None:
        raise RuntimeError(f"failed to open BigWig for writing: {path}")
    try:
        bw.addHeader(list(chroms.items()))
        if entries:
            c_list = [e[0] for e in entries]
            s_list = [e[1] for e in entries]
            e_list = [e[2] for e in entries]
            v_list = [float(e[3]) for e in entries]
            bw.addEntries(c_list, s_list, ends=e_list, values=v_list)
    finally:
        bw.close()
    return path


def _write_fasta(path: Path, records: dict[str, str]) -> Path:
    with path.open("w", encoding="ascii") as handle:
        for name, seq in records.items():
            handle.write(f">{name}\n{seq}\n")
    pysam.faidx(str(path))
    return path


def _write_sample_parquet(
    path: Path,
    contigs: list[str],
    starts: list[int],
    ends: list[int],
    counts: list[int],
    contig_lengths: dict[str, int],
    bin_size: int = 100,
) -> Path:
    return write_synthetic_sample(
        path,
        counts,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
    )


class TestFeaturesCovariates(unittest.TestCase):
    def test_validate_min_mappability(self) -> None:
        validate_min_mappability(0.5)
        validate_min_mappability(0.0)
        validate_min_mappability(1.0)
        with self.assertRaises(CovariateError):
            validate_min_mappability(-0.1)
        with self.assertRaises(CovariateError):
            validate_min_mappability(1.1)
        with self.assertRaises(CovariateError):
            validate_min_mappability(True)

    def test_mask_covariates(self) -> None:
        usable = np.array([True, True, True, True])
        gc = np.array([0.4, 0.5, float("nan"), 0.6])
        mappability = np.array([0.9, 0.3, 0.9, 0.8])
        n_mask = np.array([False, False, False, True])

        masked = mask_covariates(
            usable,
            gc=gc,
            mappability=mappability,
            n_mask=n_mask,
            min_mappability=DEFAULT_MIN_MAPPABILITY,
        )
        self.assertTrue(masked[0])
        self.assertFalse(masked[1])
        self.assertFalse(masked[2])
        self.assertFalse(masked[3])

    def test_covariate_features_and_poisson_trend(self) -> None:
        usable = np.array([True, True, True, True, False])
        gc = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
        mappability = np.array([0.8, 0.85, 0.9, 0.95, 0.5])

        features = covariate_features(usable, gc=gc, mappability=mappability)
        self.assertIsNotNone(features)
        if features is None:
            self.fail("expected non-None covariate features")
        self.assertEqual(features.shape, (4, 2))
        np.testing.assert_allclose(np.mean(features, axis=0), [0.0, 0.0], atol=1e-10)

        usable_10 = np.ones(10, dtype=np.bool_)
        gc_10 = np.linspace(0.3, 0.7, 10)
        counts_10 = (100.0 * (1.0 + 2.0 * gc_10)).astype(np.float64)
        design = build_exogenous_covariate_basis(usable_10, bin_size=5000, gc=gc_10)
        trends = fit_poisson_covariate_trends(
            counts_10,
            np.array([1.0], dtype=np.float64),
            np.ones(10, dtype=np.float64),
            usable_10,
            design,
        )[:, 0]
        self.assertEqual(trends.shape, (10,))
        self.assertLess(np.std(counts_10 / trends), np.std(counts_10))

    def test_gc_override_without_quantiles_clears_stored_fasta_quantiles(self) -> None:
        n = 40
        contigs = ("chr1",) * 20 + ("chr2",) * 20
        starts = np.tile(np.arange(20, dtype=np.int64) * 50_000, 2)
        gc_new = np.linspace(0.35, 0.65, n, dtype=np.float64)
        counts = 500.0 * np.exp(1.8 * (gc_new - 0.50))
        stale_gc_q = np.zeros((n, NUM_GC_QUANTILES), dtype=np.float32)
        stale_gc_q[:, 0] = 1.0
        pids = np.zeros((n, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pids[:20, 0] = 1
        pids[20:, 0] = 0
        pw = np.zeros((n, NUM_PROTOTYPES_PER_BIN), dtype=np.float16)
        pw[:, 0] = np.float16(1.0)

        model = BranchModel(
            branch=BRANCH_AUTOSOMAL,
            contigs=contigs,
            starts=starts,
            ends=starts + 50_000,
            counts=counts,
            masked=np.zeros(n, dtype=np.bool_),
            pca_mean=np.ones(n, dtype=np.float64),
            pca_loadings=np.empty((0, n), dtype=np.float32),
            proto_ids=pids,
            proto_weights=pw,
            mean_distance=np.zeros(n, dtype=np.float32),
            null_mean=np.zeros(n, dtype=np.float64),
            bio_dispersion=np.zeros(n, dtype=np.float64),
            gc=np.full(n, 0.50, dtype=np.float64),
            gc_quantiles=stale_gc_q,
            bin_size=50_000,
        )
        res = normalize_sample_branch(
            counts,
            np.ones(n, dtype=np.float32),
            model,
            mask_repeats=1,
            min_reference_bins=1,
            gc_override=gc_new,
        )
        self.assertLess(float(np.std(res.ratios)), 0.05)


class TestEndToEndCovariatePipeline(unittest.TestCase):
    def test_cohort_reference_and_predict_with_gc_and_mappability(self) -> None:
        rng = np.random.default_rng(7)
        n_per_contig = 25
        bin_size = 100
        contig_names = ("chr1", "chr2", "chr3", "chr4")
        contigs = [c for c in contig_names for _ in range(n_per_contig)]
        starts = [i * bin_size for i in range(n_per_contig)] * len(contig_names)
        ends = [(i + 1) * bin_size for i in range(n_per_contig)] * len(contig_names)
        lengths = {c: n_per_contig * bin_size for c in contig_names}
        n_bins = len(contigs)

        gc_vals = rng.uniform(0.32, 0.62, size=n_bins)
        map_vals = rng.uniform(0.75, 1.0, size=n_bins)
        map_vals[0] = 0.2

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bw_map = _write_bigwig(
                root / "map.bw",
                lengths,
                [
                    (contigs[i], starts[i], ends[i], float(map_vals[i]))
                    for i in range(n_bins)
                ],
            )
            bed_gc = root / "gc.bed"
            with bed_gc.open("w", encoding="utf-8") as handle:
                for i in range(n_bins):
                    handle.write(
                        f"{contigs[i]}\t{starts[i]}\t{ends[i]}\t{gc_vals[i]:.6f}\n"
                    )

            sample_paths: list[Path] = []
            g_z = (gc_vals - np.mean(gc_vals)) / np.std(gc_vals)
            for s_idx in range(8):
                slope = rng.uniform(-0.25, 0.25)
                quad = rng.uniform(-0.15, 0.05)
                bias = np.maximum(
                    0.2,
                    (1.0 + slope * g_z + quad * (g_z**2)) * (0.6 + 0.4 * map_vals),
                )
                noise = rng.normal(1.0, 0.015, size=n_bins)
                counts = np.round(1000.0 * bias * noise).astype(int)
                sample_paths.append(
                    _write_sample_parquet(
                        root / f"ctrl_{s_idx}.ccnv",
                        contigs,
                        starts,
                        ends,
                        counts.tolist(),
                        lengths,
                        bin_size=bin_size,
                    )
                )

            ref_prefix = root / "ref_cov"
            run_reference(
                ReferenceConfig(
                    entries=tuple(SamplesheetEntry(filepath=p) for p in sample_paths),
                    prefix=ref_prefix,
                    pca_components=2,
                    min_cohort_size=4,
                    mappability=bw_map,
                    gc=bed_gc,
                    min_mappability=0.5,
                )
            )

            ref_table = pq.read_table(parquet_path(ref_prefix))
            self.assertIn("gc", ref_table.column_names)
            self.assertIn("mappability", ref_table.column_names)
            self.assertTrue(ref_table.column("masked").to_pylist()[0])
            model_meta = decode_metadata(ref_table.schema.metadata)["model"]
            self.assertIsInstance(model_meta, dict)
            if not isinstance(model_meta, dict):
                self.fail("expected model metadata dictionary")
            self.assertTrue(model_meta["gc_corrected"])
            self.assertTrue(model_meta["mappability_corrected"])
            self.assertAlmostEqual(float(model_meta["min_mappability"]), 0.5)

            with open(root / "ref_cov_qc.json", encoding="utf-8") as f:
                qc_json = json.load(f)
            self.assertTrue(qc_json["parameters"]["gc_corrected"])
            self.assertTrue(qc_json["parameters"]["mappability_corrected"])

            test_bias = (1.0 + 0.30 * g_z - 0.12 * (g_z**2)) * (0.6 + 0.4 * map_vals)
            test_cn = np.ones(n_bins, dtype=np.float64)
            test_cn[10:20] = 1.5
            test_counts = np.round(1000.0 * test_bias * test_cn).astype(int)
            test_sample = _write_sample_parquet(
                root / "test_sample.ccnv",
                contigs,
                starts,
                ends,
                test_counts.tolist(),
                lengths,
                bin_size=bin_size,
            )

            outputs = run_predict(
                PredictConfig(
                    sample=test_sample,
                    reference=parquet_path(ref_prefix),
                    prefix=root / "pred",
                    penalty=12.0,
                    plot=False,
                    vcf=False,
                    min_reference_bins=3,
                )
            )
            self.assertTrue(outputs.bins[0].masked[0])
            gains = [s for s in outputs.segments if s.call == "gain"]
            self.assertTrue(
                any(
                    s.contig == "chr1" and s.start == 1000 and s.end == 2000
                    for s in gains
                ),
                msg=f"segments={outputs.segments}",
            )

    def test_flat_fasta_reference_uses_gc_and_mappability_for_matching_and_predict(
        self,
    ) -> None:
        rng = np.random.default_rng(99)
        n_per_contig = 25
        bin_size = 100
        contig_names = ("chr1", "chr2", "chr3", "chr4")
        contigs = [c for c in contig_names for _ in range(n_per_contig)]
        starts = [i * bin_size for i in range(n_per_contig)] * len(contig_names)
        ends = [(i + 1) * bin_size for i in range(n_per_contig)] * len(contig_names)
        lengths = {c: n_per_contig * bin_size for c in contig_names}
        n_bins = len(contigs)

        gc_fracs = rng.uniform(0.30, 0.70, size=n_per_contig)
        seqs: dict[str, list[str]] = {c: [] for c in contig_names}
        for c_idx, cname in enumerate(contig_names):
            for b_idx in range(n_per_contig):
                if c_idx == 0 and b_idx == 0:
                    seqs[cname].append("N" * bin_size)
                    continue
                n_gc = round(float(gc_fracs[b_idx]) * bin_size)
                n_at = bin_size - n_gc
                seqs[cname].append("G" * n_gc + "A" * n_at)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = _write_fasta(
                root / "genome.fa",
                {k: "".join(v) for k, v in seqs.items()},
            )
            map_base = rng.uniform(0.65, 1.0, size=n_per_contig)
            map_vals = np.tile(map_base, len(contig_names))
            map_vals[1] = 0.1
            bw_map = _write_bigwig(
                root / "map.bw",
                lengths,
                [
                    (contigs[i], starts[i], ends[i], float(map_vals[i]))
                    for i in range(n_bins)
                ],
            )

            ref_prefix = root / "flat_cov"
            res = CliRunner().invoke(
                app,
                [
                    "reference",
                    "--fasta",
                    str(fasta),
                    "--prefix",
                    str(ref_prefix),
                    "--bin-size",
                    str(bin_size),
                    "--min-cohort-size",
                    "2",
                    "--mappability",
                    str(bw_map),
                    "--min-mappability",
                    "0.5",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)

            ref_table = pq.read_table(parquet_path(ref_prefix))
            masked_col = ref_table.column("masked").to_pylist()
            self.assertTrue(masked_col[0])
            self.assertTrue(masked_col[1])
            self.assertFalse(masked_col[2])

            gc_col = np.asarray(ref_table.column("gc").to_pylist(), dtype=np.float64)
            g_z = (gc_fracs - np.mean(gc_fracs)) / np.std(gc_fracs)
            full_gz = np.tile(g_z, len(contig_names))
            wave = np.maximum(
                0.2,
                (1.0 + 0.25 * full_gz - 0.10 * (full_gz**2)) * (0.7 + 0.3 * map_vals),
            )
            true_cn = np.ones(n_bins, dtype=np.float64)
            true_cn[10:18] = 1.5
            counts = np.round(1000.0 * wave * true_cn * rng.normal(1.0, 0.01, n_bins))
            sample_path = _write_sample_parquet(
                root / "sample.ccnv",
                contigs,
                starts,
                ends,
                counts.astype(int).tolist(),
                lengths,
                bin_size=bin_size,
            )

            outputs = run_predict(
                PredictConfig(
                    sample=sample_path,
                    reference=parquet_path(ref_prefix),
                    prefix=root / "flat_pred",
                    penalty=12.0,
                    plot=False,
                    vcf=False,
                    min_reference_bins=2,
                    ratio_threshold=0.35,
                )
            )
            self.assertGreater(int(np.count_nonzero(gc_col >= 0)), 80)
            gains = [s for s in outputs.segments if s.call == "gain"]
            self.assertTrue(
                any(
                    s.contig == "chr1" and s.start == 1000 and s.end == 1800
                    for s in gains
                ),
                msg=f"segments={outputs.segments}",
            )

    def test_predict_cli_covariate_override_flags(self) -> None:
        rng = np.random.default_rng(1234)
        n_per_contig = 25
        bin_size = 100
        contig_names = ("chr1", "chr2", "chr3", "chr4")
        contigs = [c for c in contig_names for _ in range(n_per_contig)]
        starts = [i * bin_size for i in range(n_per_contig)] * len(contig_names)
        ends = [(i + 1) * bin_size for i in range(n_per_contig)] * len(contig_names)
        lengths = {c: n_per_contig * bin_size for c in contig_names}
        n_bins = len(contigs)

        gc_vals = rng.uniform(0.32, 0.62, size=n_bins)
        map_vals = rng.uniform(0.75, 1.0, size=n_bins)
        map_vals[0] = 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bw_map = _write_bigwig(
                root / "map.bw",
                lengths,
                [
                    (contigs[i], starts[i], ends[i], float(map_vals[i]))
                    for i in range(n_bins)
                ],
            )
            bed_gc = root / "gc.bed"
            with bed_gc.open("w", encoding="utf-8") as handle:
                for i in range(n_bins):
                    handle.write(
                        f"{contigs[i]}\t{starts[i]}\t{ends[i]}\t{gc_vals[i]:.6f}\n"
                    )

            sample_paths: list[Path] = []
            for s_idx in range(6):
                counts = np.round(1000.0 * rng.normal(1.0, 0.015, size=n_bins)).astype(
                    int
                )
                sample_paths.append(
                    _write_sample_parquet(
                        root / f"ctrl_{s_idx}.ccnv",
                        contigs,
                        starts,
                        ends,
                        counts.tolist(),
                        lengths,
                        bin_size=bin_size,
                    )
                )

            ref_prefix = root / "ref_no_cov"
            run_reference(
                ReferenceConfig(
                    entries=tuple(SamplesheetEntry(filepath=p) for p in sample_paths),
                    prefix=ref_prefix,
                    pca_components=2,
                    min_cohort_size=4,
                    qc_report=False,
                )
            )

            g_z = (gc_vals - np.mean(gc_vals)) / np.std(gc_vals)
            test_bias = (1.0 + 0.30 * g_z - 0.12 * (g_z**2)) * (0.6 + 0.4 * map_vals)
            test_cn = np.ones(n_bins, dtype=np.float64)
            test_cn[8:16] = 1.5
            test_counts = np.round(1000.0 * test_bias * test_cn).astype(int)
            test_sample = _write_sample_parquet(
                root / "test.ccnv",
                contigs,
                starts,
                ends,
                test_counts.tolist(),
                lengths,
                bin_size=bin_size,
            )

            pred_prefix = root / "pred_override"
            res = CliRunner().invoke(
                app,
                [
                    "predict",
                    str(test_sample),
                    str(parquet_path(ref_prefix)),
                    "--prefix",
                    str(pred_prefix),
                    "--mappability",
                    str(bw_map),
                    "--gc",
                    str(bed_gc),
                    "--min-mappability",
                    "0.5",
                    "--penalty",
                    "0.1",
                    "--no-plot",
                    "--no-vcf",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            with gzip.open(f"{pred_prefix}_bins.bed.gz", "rt") as h:
                bin_rows = h.read().splitlines()
            self.assertIn("true", bin_rows[1])
            with gzip.open(f"{pred_prefix}_calls.bed.gz", "rt") as h:
                call_rows = h.read().splitlines()
            self.assertTrue(any("gain" in row for row in call_rows[1:]))

    def test_reference_combines_fasta_n_mask_with_gc_track(self) -> None:
        contig_names = ("chr1", "chr2")
        n_per = 10
        bin_size = 100
        contigs = [c for c in contig_names for _ in range(n_per)]
        starts = [i * bin_size for i in range(n_per)] * len(contig_names)
        ends = [(i + 1) * bin_size for i in range(n_per)] * len(contig_names)
        n_bins = len(contigs)

        seqs = {
            "chr1": "N" * bin_size + "A" * (bin_size * (n_per - 1)),
            "chr2": "A" * (bin_size * n_per),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta_path = _write_fasta(root / "genome.fa", seqs)

            bed_gc = root / "gc.bed"
            with bed_gc.open("w", encoding="utf-8") as handle:
                for i in range(n_bins):
                    handle.write(f"{contigs[i]}\t{starts[i]}\t{ends[i]}\t0.45\n")

            ref_prefix = root / "ref_fasta_n"
            run_reference(
                ReferenceConfig(
                    fasta=fasta_path,
                    gc=bed_gc,
                    bin_size=bin_size,
                    prefix=ref_prefix,
                    min_cohort_size=2,
                    pca_components=1,
                )
            )

            ref_table = pq.read_table(parquet_path(ref_prefix))
            masked = ref_table.column("masked").to_pylist()
            gc_vals = ref_table.column("gc").to_pylist()

            self.assertTrue(masked[0])
            self.assertFalse(masked[1])
            self.assertAlmostEqual(gc_vals[0], 0.45, places=4)
            self.assertAlmostEqual(gc_vals[1], 0.45, places=4)

    def test_defect4_single_pass_gc_track_across_multi_branch_predict(self) -> None:
        contig_lengths = {
            "chr1": 100_000,
            "chr2": 100_000,
            "chrX": 100_000,
            "chrY": 100_000,
        }
        bin_size = 5000
        n_tot = 80
        rng = np.random.default_rng(404)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta_path = root / "ref.fa"
            seq_chunk = ("ACGTGCGC" * 12_500)[:100_000]
            fasta_path.write_text(
                f">chr1\n{seq_chunk}\n>chr2\n{seq_chunk}\n"
                f">chrX\n{seq_chunk}\n>chrY\n{seq_chunk}\n",
                encoding="utf-8",
            )

            contigs_80 = [c for c in contig_lengths for _ in range(20)]
            starts_80 = [i * bin_size for i in range(20)] * 4
            ends_80 = [(i + 1) * bin_size for i in range(20)] * 4
            sample_paths: list[Path] = []
            for i in range(6):
                is_male = i >= 3
                c_arr = rng.poisson(120, size=n_tot).astype(np.uint32)
                if is_male:
                    c_arr[40:60] = rng.poisson(60, size=20).astype(np.uint32)
                    c_arr[60:] = rng.poisson(60, size=20).astype(np.uint32)
                else:
                    c_arr[60:] = 0
                sp = root / f"g_{i}.ccnv"
                _write_sample_parquet(
                    sp,
                    contigs_80,
                    starts_80,
                    ends_80,
                    c_arr.tolist(),
                    contig_lengths,
                    bin_size=bin_size,
                )
                sample_paths.append(sp)

            ref_prefix = root / "germ_ref"
            run_reference(
                ReferenceConfig(
                    entries=tuple(
                        SamplesheetEntry(filepath=p, sex="M" if idx >= 3 else "F")
                        for idx, p in enumerate(sample_paths)
                    ),
                    prefix=ref_prefix,
                    bin_size=bin_size,
                    min_cohort_size=4,
                    min_sex_group_size=2,
                    qc_report=False,
                )
            )
            with patch.object(
                ccnv.predict, "load_gc_track", wraps=load_gc_track
            ) as spy_gc:
                run_predict(
                    PredictConfig(
                        sample=sample_paths[0],
                        reference=parquet_path(ref_prefix),
                        prefix=root / "pred_f",
                        sex="female",
                        gc=fasta_path,
                        plot=False,
                        min_reference_bins=1,
                    )
                )
                self.assertEqual(spy_gc.call_count, 1)

    def test_gc_quantiles_roundtrip_and_flat_fasta_distance_weights(self) -> None:
        rng = np.random.default_rng(31415)
        bin_size = 50_000
        bins_per_contig = 24
        contig_names = ("chr1", "chr2", "chr3")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fasta_path = tmp_path / "ref.fa"
            lines: list[str] = []
            for c_idx, cname in enumerate(contig_names):
                lines.append(f">{cname}")
                chunks: list[str] = []
                for b in range(bins_per_contig):
                    gc_prob = 0.30 + 0.35 * (((b + 3 * c_idx) % 8) / 7.0)
                    bases = rng.choice(
                        np.array(["A", "T", "G", "C"]),
                        size=bin_size,
                        p=[
                            0.5 * (1.0 - gc_prob),
                            0.5 * (1.0 - gc_prob),
                            0.5 * gc_prob,
                            0.5 * gc_prob,
                        ],
                    )
                    chunks.append("".join(bases.tolist()))
                seq = "".join(chunks)
                for i in range(0, len(seq), 80):
                    lines.append(seq[i : i + 80])
            fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            flat_cfg = ReferenceConfig(
                fasta=fasta_path,
                bin_size=bin_size,
                prefix=tmp_path / "flat_ref",
                min_cohort_size=2,
                qc_report=False,
            )
            ref_path = run_reference(flat_cfg)
            loaded_tbl, loaded_meta = read_reference_parquet(ref_path)
            flat_model = build_branch_model(loaded_tbl, BRANCH_AUTOSOMAL, loaded_meta)
            self.assertTrue(flat_model.is_flat)
            self.assertIsNotNone(flat_model.gc_quantiles)
            if flat_model.gc_quantiles is not None:
                self.assertEqual(
                    flat_model.gc_quantiles.shape,
                    (len(contig_names) * bins_per_contig, 8),
                )

            varied_dist = np.linspace(
                0.05, 2.0, flat_model.counts.size, dtype=np.float32
            )
            flat_model_varied = replace(
                flat_model, mean_distance=varied_dist, is_flat=True
            )
            exposure = compute_bin_exposure(
                flat_model.starts, flat_model.ends, bin_size
            )
            sample_counts = rng.poisson(35.0, size=flat_model.counts.size).astype(
                np.float64
            )
            norm_flat = normalize_sample_branch(
                sample_counts, exposure, flat_model_varied
            )
            self.assertGreater(
                float(norm_flat.weights[0]), float(norm_flat.weights[-1]) * 5.0
            )


if __name__ == "__main__":
    unittest.main()
