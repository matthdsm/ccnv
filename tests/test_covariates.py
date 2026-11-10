import gzip
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyBigWig
import pysam
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.features.covariates import (
    correct_covariate_bias,
    covariate_features,
    mask_covariates,
    validate_min_mappability,
)
from ccnv.io.tracks import (
    _bin_weighted_mean,
    load_gc_track,
    load_mappability_track,
)
from ccnv.models import (
    CovariateError,
    PredictConfig,
    ReadFilters,
    ReferenceConfig,
    parquet_path,
)
from ccnv.models.schemas import (
    SAMPLE_SCHEMA,
    ArtifactKind,
    decode_metadata,
    metadata_for,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference


def _write_bigwig(
    path: Path,
    chroms: dict[str, int],
    entries: list[tuple[str, int, int, float]],
) -> Path:
    bw = pyBigWig.open(str(path), "w")
    assert bw is not None
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
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": bin_size,
            "source_identity": {"id": path.stem},
            "reference_identity": {"name": "test"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )
    n_bins = len(contigs)
    table = pa.table(
        {
            "contig": contigs,
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "count": pa.array(counts, type=pa.uint64()),
            "fragment_lengths": pa.array([[]] * n_bins, type=pa.list_(pa.uint16())),
            "endpoint_count": pa.array(
                np.zeros(n_bins, dtype=np.uint32), type=pa.uint32()
            ),
            "frag_gc_sum": pa.array(
                np.zeros(n_bins, dtype=np.float32), type=pa.float32()
            ),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )
    pq.write_table(table, path)
    return path


class TestTrackLoadingAndMasking(unittest.TestCase):
    def test_bin_weighted_mean_full_bin_vs_covered_len(self) -> None:
        starts = np.array([0, 100, 200], dtype=np.int64)
        ends = np.array([100, 200, 300], dtype=np.int64)
        # Interval covers only 40 bp of bin 0 (value 0.5) and 0 bp of bin 1, 2
        iv_s = np.array([0], dtype=np.int64)
        iv_e = np.array([40], dtype=np.int64)
        iv_v = np.array([0.5], dtype=np.float64)

        # Mappability track: divide by full bin length, uncovered bins at 0.0
        map_mean = _bin_weighted_mean(
            starts, ends, iv_s, iv_e, iv_v, full_bin_denominator=True
        )
        # (40 * 0.5) / 100 = 0.2; bin 1 and 2 have no coverage -> 0.0
        np.testing.assert_allclose(map_mean, [0.2, 0.0, 0.0], atol=1e-6)

        # GC track: divide by covered length, uncovered bins at NaN
        gc_mean = _bin_weighted_mean(
            starts, ends, iv_s, iv_e, iv_v, full_bin_denominator=False
        )
        # (40 * 0.5) / 40 = 0.5; bin 1 and 2 have no coverage -> NaN
        self.assertAlmostEqual(gc_mean[0], 0.5)
        self.assertTrue(np.isnan(gc_mean[1]))
        self.assertTrue(np.isnan(gc_mean[2]))

    def test_load_mappability_bigwig_and_chr_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bw_path = _write_bigwig(
                Path(tmpdir) / "map.bw",
                {"1": 300, "2": 400},
                [
                    ("1", 0, 50, 1.0),
                    ("1", 50, 100, 0.5),
                    ("1", 100, 200, 0.2),
                    ("1", 200, 300, 0.9),
                    ("2", 0, 200, 0.8),
                    ("2", 200, 250, 0.6),  # Partially covers bin 200..300
                ],
            )
            contigs = ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2", "chr2"]
            starts = np.array([0, 100, 200, 0, 100, 200, 300], dtype=np.int64)
            ends = np.array([100, 200, 300, 100, 200, 300, 400], dtype=np.int64)

            values = load_mappability_track(bw_path, contigs, starts, ends)
            # Bin 5 (chr2:200..300): (50 * 0.6) / 100 = 0.3
            # Bin 6 (chr2:300..400): uncovered -> 0.0
            np.testing.assert_allclose(
                values, [0.75, 0.2, 0.9, 0.8, 0.8, 0.3, 0.0], atol=1e-6
            )

            usable = np.ones(7, dtype=np.bool_)
            masked = mask_covariates(usable, mappability=values, min_mappability=0.5)
            np.testing.assert_array_equal(
                masked, [True, False, True, True, True, False, False]
            )

    def test_load_gc_from_fasta_bed_and_bigwig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq_chr1 = (
                ("G" * 40 + "A" * 60) + ("C" * 70 + "T" * 30) + ("N" * 80 + "G" * 20)
            )
            fasta = _write_fasta(root / "genome.fa", {"chr1": seq_chr1})
            contigs = ["chr1", "chr1", "chr1"]
            starts = np.array([0, 100, 200], dtype=np.int64)
            ends = np.array([100, 200, 300], dtype=np.int64)

            gc_fa, n_mask_fa = load_gc_track(fasta, contigs, starts, ends)
            np.testing.assert_allclose(gc_fa[:2], [0.4, 0.7], atol=1e-6)
            self.assertTrue(np.isnan(gc_fa[2]))
            np.testing.assert_array_equal(n_mask_fa, [False, False, True])

            bed_gz = root / "gc.bed.gz"
            with gzip.open(bed_gz, "wt", encoding="utf-8") as handle:
                handle.write("#chrom\tstart\tend\tgc\n")
                handle.write("chr1\t0\t100\t40.0\n")
                handle.write("chr1\t100\t200\t70.0\n")
            gc_bed, n_mask_bed = load_gc_track(bed_gz, contigs, starts, ends)
            np.testing.assert_allclose(gc_bed[:2], [0.4, 0.7], atol=1e-6)
            self.assertTrue(np.isnan(gc_bed[2]))
            np.testing.assert_array_equal(n_mask_bed, [False, False, True])

            bw_gc = _write_bigwig(
                root / "gc.bw",
                {"chr1": 300},
                [("chr1", 0, 100, 0.4), ("chr1", 100, 200, 0.7)],
            )
            gc_bw, n_mask_bw = load_gc_track(bw_gc, contigs, starts, ends)
            np.testing.assert_allclose(gc_bw[:2], [0.4, 0.7], atol=1e-6)
            np.testing.assert_array_equal(n_mask_bw, [False, False, True])

    def test_validate_min_mappability_rejects_invalid_values(self) -> None:
        for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(CovariateError):
                validate_min_mappability(bad)  # type: ignore[arg-type]


class TestCovariateRegression(unittest.TestCase):
    def test_correct_covariate_bias_removes_wave_and_preserves_cnv(self) -> None:
        rng = np.random.default_rng(42)
        n_bins = 200
        gc = rng.uniform(0.30, 0.65, size=n_bins)
        mappability = rng.uniform(0.65, 1.0, size=n_bins)
        usable = np.ones(n_bins, dtype=np.bool_)

        g_centered = (gc - 0.45) / 0.10
        wave = (
            1.0 + 0.25 * g_centered - 0.15 * (g_centered**2) + 0.05 * (g_centered**3)
        ) * (0.7 + 0.3 * mappability)

        true_cn = np.ones(n_bins, dtype=np.float64)
        true_cn[80:95] = 1.5  # Planted focal gain
        y = true_cn * wave
        y /= np.sum(y)

        corrected = correct_covariate_bias(y, usable, gc=gc, mappability=mappability)

        neutral = np.ones(n_bins, dtype=np.bool_)
        neutral[80:95] = False
        raw_cv = float(np.std(y[neutral]) / np.mean(y[neutral]))
        corr_cv = float(np.std(corrected[neutral]) / np.mean(corrected[neutral]))
        self.assertLess(corr_cv, raw_cv * 0.1)

        gain_ratio = float(np.median(corrected[80:95]) / np.median(corrected[neutral]))
        self.assertAlmostEqual(gain_ratio, 1.5, delta=0.03)

    def test_covariate_features_standardizes_columns(self) -> None:
        usable = np.array([True, True, True, False, True])
        gc = np.array([0.3, 0.4, 0.5, 0.9, 0.6])
        mappability = np.array([0.7, 0.8, 0.9, 0.1, 1.0])
        feats = covariate_features(usable, gc=gc, mappability=mappability)
        self.assertIsNotNone(feats)
        assert feats is not None
        self.assertEqual(feats.shape, (4, 2))
        np.testing.assert_allclose(np.mean(feats, axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.std(feats, axis=0), [1.0, 1.0], atol=1e-12)

    def test_build_joint_covariate_basis_captures_gc_mappability_interaction(
        self,
    ) -> None:
        from ccnv.features.covariates import correct_covariate_bias
        from ccnv.stats.regression import build_joint_covariate_basis

        rng = np.random.default_rng(42)
        n = 200
        gc = rng.uniform(0.25, 0.75, size=n)
        mapp = rng.uniform(0.55, 1.0, size=n)
        basis = build_joint_covariate_basis(gc, mapp)
        self.assertIsNotNone(basis)

        # Simulate coupled non-linear GC + mappability + interaction bias with 10% CNV outliers
        z_gc = (gc - np.mean(gc)) / np.std(gc)
        z_map = (mapp - np.mean(mapp)) / np.std(mapp)
        true_bias = np.exp(0.25 * np.sin(z_gc) + 0.15 * z_map + 0.12 * (z_gc * z_map))
        raw = 100.0 * true_bias
        raw[:15] *= 2.0  # 15 amplified CNV bins
        usable = np.ones(n, dtype=np.bool_)

        corrected = correct_covariate_bias(raw, usable, gc=gc, mappability=mapp)
        diploid_cv_before = np.std(raw[15:]) / np.mean(raw[15:])
        diploid_cv_after = np.std(corrected[15:]) / np.mean(corrected[15:])
        self.assertLess(diploid_cv_after, diploid_cv_before * 0.25)
        # CNV gain ratio on bins 0..14 must be preserved (~2.0x relative to diploid median)
        self.assertAlmostEqual(
            float(np.median(corrected[:15]) / np.median(corrected[15:])),
            2.0,
            delta=0.15,
        )


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
        map_vals[0] = 0.2  # Low mappability bin -> should be masked

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
                    samples=tuple(sample_paths),
                    prefix=ref_prefix,
                    pca_components=2,
                    nearest_bins=15,
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
            assert isinstance(model_meta, dict)
            self.assertTrue(model_meta["gc_corrected"])
            self.assertTrue(model_meta["mappability_corrected"])
            self.assertAlmostEqual(float(model_meta["min_mappability"]), 0.5)

            with open(root / "ref_cov_qc.json", encoding="utf-8") as f:
                qc_json = json.load(f)
            self.assertTrue(qc_json["parameters"]["gc_corrected"])
            self.assertTrue(qc_json["parameters"]["mappability_corrected"])

            test_bias = (1.0 + 0.30 * g_z - 0.12 * (g_z**2)) * (0.6 + 0.4 * map_vals)
            test_cn = np.ones(n_bins, dtype=np.float64)
            test_cn[10:20] = 1.5  # Gain on chr1:1000-2000
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
                    penalty=0.1,
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
                    seqs[cname].append("N" * bin_size)  # N-masked bin
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
            map_vals[1] = 0.1  # Low mappability bin on chr1
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
                    str(fasta),
                    "--prefix",
                    str(ref_prefix),
                    "--bin-size",
                    str(bin_size),
                    "--min-cohort-size",
                    "2",
                    "--nearest-bins",
                    "6",
                    "--mappability",
                    str(bw_map),
                    "--min-mappability",
                    "0.5",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)

            ref_table = pq.read_table(parquet_path(ref_prefix))
            masked_col = ref_table.column("masked").to_pylist()
            self.assertTrue(masked_col[0])  # N-masked bin
            self.assertTrue(masked_col[1])  # Low-mappability bin
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
                    penalty=0.1,
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
        map_vals[0] = 0.1  # Masked only when predict overrides mappability

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
                    samples=tuple(sample_paths),
                    prefix=ref_prefix,
                    pca_components=2,
                    nearest_bins=15,
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
            self.assertIn("true", bin_rows[1])  # Bin 0 masked by override
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

        # FASTA has N-gap on bin 0 of chr1
        seqs = {
            "chr1": "N" * bin_size + "A" * (bin_size * (n_per - 1)),
            "chr2": "A" * (bin_size * n_per),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta_path = _write_fasta(root / "genome.fa", seqs)

            # GC BED gives valid GC (0.45) for ALL bins (including bin 0)
            bed_gc = root / "gc.bed"
            with bed_gc.open("w", encoding="utf-8") as handle:
                for i in range(n_bins):
                    handle.write(f"{contigs[i]}\t{starts[i]}\t{ends[i]}\t0.45\n")

            ref_prefix = root / "ref_fasta_n"
            run_reference(
                ReferenceConfig(
                    samples=(),
                    fasta=fasta_path,
                    gc=bed_gc,
                    bin_size=bin_size,
                    prefix=ref_prefix,
                    min_cohort_size=2,
                    nearest_bins=4,
                    pca_components=1,
                )
            )

            ref_table = pq.read_table(parquet_path(ref_prefix))
            masked = ref_table.column("masked").to_pylist()
            gc_vals = ref_table.column("gc").to_pylist()

            # Bin 0 masked due to FASTA N-mask combining with GC track
            self.assertTrue(masked[0])
            self.assertFalse(masked[1])
            # GC column sourced from GC BED (0.45)
            self.assertAlmostEqual(gc_vals[0], 0.45, places=4)
            self.assertAlmostEqual(gc_vals[1], 0.45, places=4)


if __name__ == "__main__":
    unittest.main()
