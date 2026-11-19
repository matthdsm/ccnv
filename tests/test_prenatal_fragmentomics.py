"""Unit and integration tests for prenatal fragmentomics, Fisher weighting, and FF_size."""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from helpers import make_sample_table
from numpy.typing import NDArray
from scipy.stats import norm

import ccnv.features.fragmentomics as ff_mod
import ccnv.stats.fragmentomics as sf
from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.binning import bin_intervals
from ccnv.features.cohort import build_fisher_weighted_cohort, load_and_assemble_cohort
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    extract_sample_fragmentomics,
    resolve_fisher_moment_weights,
)
from ccnv.features.normalization import BranchModel, normalize_sample_branch
from ccnv.features.prenatal import (
    _compute_effective_ff,
    _evaluate_sca_and_active_ff,
    fit_prenatal_models,
    predict_prenatal,
)
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    annotate_prenatal_origin_and_lod,
    annotate_sample_prenatal_segments,
    classify_segment_origin_5state,
)
from ccnv.io.parquet import write_sample_parquet
from ccnv.models import (
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    SCHEMA_VERSION,
    FetalFractionResult,
    FragmentomicsMetadata,
    PredictConfig,
    PrenatalModelMetadata,
    ReferenceConfig,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SCAResult,
    SegmentResult,
    assemble_reference_table,
    parquet_path,
)
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.stats.fragmentomics import (
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
    compute_band_fisher_weights,
)
from ccnv.stats.prenatal_math import classify_sca_ray_4d


def _fetal_maternal_densities() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
    p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
    p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
    return p_fet / np.sum(p_fet), p_mat / np.sum(p_mat)


def _make_dummy_prenatal_metadata(**overrides: object) -> PrenatalModelMetadata:
    defaults: dict[str, object] = {
        "prenatal_bin_size": 100000,
        "yfrac_threshold": 0.001,
        "ffy_slope": 1.0,
        "olm_weights": [],
        "olm_bias": 0.0,
        "female_chrx_ratio_mean": 1.0,
        "female_chrx_ratio_std": 0.01,
        "ffy_std": 0.01,
        "ffa_std": 0.02,
        "ff_size_std": 0.018,
        **overrides,
    }
    return PrenatalModelMetadata.from_dict(defaults)


def _frag_meta(
    hist_auto: list[int],
    hist_chry: list[int],
    hist_chrx: list[int],
    motifs: list[int],
    *,
    paired_end: bool = True,
    shift_bp: float = 0.0,
    peak_ratio: float = 0.35,
) -> FragmentomicsMetadata:
    scalars = (paired_end, sum(hist_auto), 166.0, 143.0, shift_bp, peak_ratio)
    shape = (0.08, 0.03, 0.05, 1.2, 0.92)
    return FragmentomicsMetadata(
        *scalars, *shape, hist_auto, hist_chry, hist_chrx, motifs
    )


def _fisher_prenatal_meta() -> PrenatalModelMetadata:
    p_fet, p_mat = _fetal_maternal_densities()
    bw, mf, mm = compute_band_fisher_weights(p_fet, p_mat)
    return _make_dummy_prenatal_metadata(
        p_fetal_density=p_fet.tolist(),
        p_maternal_density=p_mat.tolist(),
        band_fisher_weights=bw.tolist(),
        mu_fetal_w=mf,
        mu_maternal_w=mm,
    )


def _make_mock_fragmentomics_metadata(
    *,
    paired_end: bool = True,
    shift_bp: float = 0.0,
    fetal_fraction: float = 0.10,
    is_male: bool = True,
    seed: int = 42,
) -> FragmentomicsMetadata:
    rng = np.random.default_rng(seed)
    p_fet, p_mat = _fetal_maternal_densities()
    p_mix = (1.0 - fetal_fraction) * p_mat + fetal_fraction * p_fet
    hist_auto = rng.multinomial(50000, p_mix / np.sum(p_mix)).tolist()
    hist_chry = (
        rng.multinomial(1000, p_fet).tolist()
        if is_male
        else rng.multinomial(20, p_mat).tolist()
    )
    hist_chrx = rng.multinomial(2000, p_mix / np.sum(p_mix)).tolist()
    peak_ratio = float(p_mix[143 - FRAG_MIN_LEN] / p_mix[166 - FRAG_MIN_LEN])
    return _frag_meta(
        hist_auto,
        hist_chry,
        hist_chrx,
        rng.poisson(100, size=256).tolist(),
        paired_end=paired_end,
        shift_bp=shift_bp,
        peak_ratio=peak_ratio,
    )


def _make_mock_sample_table(
    contigs: list[str],
    n_bins_per_contig: int = 10,
    bin_size: int = 5000,
    is_male: bool = True,
    frag_meta: FragmentomicsMetadata | None = None,
    fetal_enriched_bins: set[int] | None = None,
    fetal_fraction: float = 0.10,
) -> pa.Table:
    c_list = [c for c in contigs for _ in range(n_bins_per_contig)]
    s_list = [b * bin_size for _ in contigs for b in range(n_bins_per_contig)]
    e_list = [s + bin_size for s in s_list]
    counts: list[float] = []
    frag_lengths_list: list[list[int]] = []
    rng = np.random.default_rng(123)
    for bin_idx, contig in enumerate(c_list):
        if contig.endswith("Y"):
            counts.append(100.0 * (0.5 * fetal_fraction) if is_male else 0.0)
        elif contig.endswith("X"):
            counts.append(100.0 * (1.0 - 0.5 * fetal_fraction) if is_male else 100.0)
        else:
            counts.append(100.0)
        enriched = bool(fetal_enriched_bins and bin_idx in fetal_enriched_bins)
        pool = np.arange(120, 150) if enriched else np.arange(155, 175)
        frag_lengths_list.append(rng.choice(pool, size=100).tolist())

    n_tot = len(counts)
    cl = {c: n_bins_per_contig * bin_size for c in contigs}
    table = make_sample_table(
        counts,
        contigs=c_list,
        starts=s_list,
        ends=e_list,
        contig_lengths=cl,
        bin_size=bin_size,
        fragment_lengths=frag_lengths_list,
        endpoint_counts=[50] * n_tot,
        frag_gc_sums=[42.0] * n_tot,
    )
    if frag_meta is not None:
        meta_dict = dict(table.schema.metadata or {})
        meta_dict[b"ccnv.fragmentomics"] = json.dumps(asdict(frag_meta)).encode()
        table = table.replace_schema_metadata(meta_dict)
    return table


def _make_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: NDArray[np.uint32] | NDArray[np.float64],
    bands: NDArray[np.uint32] | None = None,
    *,
    frag_meta: FragmentomicsMetadata | None = None,
) -> SampleBuffers:
    n, n_kept = counts.size, max(1000, int(np.sum(counts)))
    qc = {
        "effective_reads_examined": n_kept,
        "effective_reads_kept": n_kept,
        "autosomal_reads_kept": n_kept,
        "sex_reads_kept": 0,
    }
    meta = SampleMetadata(
        bin_size=bin_size,
        contig_order=tuple(contig_lengths.keys()),
        contig_lengths=contig_lengths,
        filters={"min_mapping_quality": 1},
        qc=qc,
        schema_version=SCHEMA_VERSION,
        coordinate_system=COORDINATE_SYSTEM,
        fragmentomics=frag_meta,
    )
    z_u32 = np.zeros(n, dtype=np.uint32)
    b_u32 = np.asarray(bands, dtype=np.uint32) if bands is not None else None
    p_cnt = (
        np.sum(b_u32, axis=1, dtype=np.uint32)
        if b_u32 is not None
        else (np.asarray(counts, dtype=np.uint32) if frag_meta is not None else z_u32)
    )
    s_cnt = (
        np.sum(b_u32[:, :8], axis=1, dtype=np.uint32) if b_u32 is not None else z_u32
    )
    return SampleBuffers(
        meta,
        counts,
        p_cnt,
        np.full(n, 16600, dtype=np.uint64),
        s_cnt,
        z_u32.copy(),
        np.zeros(n, dtype=np.float32),
        b_u32,
    )


def _make_unlabeled_prenatal_sample(
    contig_lengths: dict[str, int], is_male: bool, ff: float, seed: int
) -> pa.Table:
    rng = np.random.default_rng(seed)
    pf, pm = _fetal_maternal_densities()
    pmix = ff * pf + (1.0 - ff) * pm
    hist_auto = rng.multinomial(60_000, pmix / np.sum(pmix)).astype(np.int64)
    hist_y = (
        rng.multinomial(max(50, int(3_000 * ff)), pf).astype(np.int64)
        if is_male
        else np.zeros(FRAG_NUM_BINS, dtype=np.int64)
    )
    motifs = np.full(256, 200, dtype=np.int64)
    lm = sf.estimate_sample_landmarks(hist_auto, motifs, paired_end=True)
    frag_meta = FragmentomicsMetadata(
        True,
        int(np.sum(hist_auto)),
        lm.chromatosome_mode_bp,
        lm.core_shoulder_bp,
        lm.sizing_shift_bp,
        lm.self_referenced_peak_ratio,
        lm.dinucleosome_fraction,
        lm.high_mw_fraction,
        lm.helical_10bp_amplitude,
        lm.helical_10bp_phase,
        lm.motif_diversity_score,
        hist_auto.tolist(),
        hist_y.tolist(),
        (hist_auto // 20).tolist(),
        motifs.tolist(),
    )
    counts_list: list[int] = []
    for contig, length in contig_lengths.items():
        n_b = bin_intervals(length, 5000)[0].size
        if contig == "chrY":
            mean_c = 100.0 * (0.5 * ff) if is_male else 0.05
        else:
            mean_c = (
                100.0 * (1.0 - 0.5 * ff) if (contig == "chrX" and is_male) else 100.0
            )
        counts_list.extend(rng.poisson(lam=mean_c, size=n_b).tolist())
    c_arr = np.asarray(counts_list, dtype=np.uint32)
    return _make_buffers(contig_lengths, 5000, c_arr, frag_meta=frag_meta).to_table()


class TestPrenatalFragmentomics(unittest.TestCase):
    def test_extract_sample_fragmentomics_and_metadata(self) -> None:
        frag_meta = _make_mock_fragmentomics_metadata()
        table = _make_mock_sample_table(["chr1", "chrY"], frag_meta=frag_meta)

        extracted = extract_sample_fragmentomics(table.schema.metadata)
        if extracted is None:
            self.fail("expected extracted fragmentomics metadata")
        self.assertTrue(extracted.paired_end)
        self.assertEqual(len(extracted.hist_autosomal), FRAG_NUM_BINS)

    def test_band_prefix_sums_query(self) -> None:
        table = _make_mock_sample_table(
            ["chr1", "chr2"], n_bins_per_contig=5, bin_size=5000
        )
        ps = BandPrefixSums.from_sample_table(table)
        if ps is None:
            self.fail("expected BandPrefixSums")

        bands = ps.query("chr1", 0, 15000)
        self.assertEqual(bands.shape, (NUM_FRAG_BANDS,))
        self.assertAlmostEqual(float(np.sum(bands)), 300.0)

        empty = ps.query("chrUn", 0, 5000)
        self.assertAlmostEqual(float(np.sum(empty)), 0.0)

    def test_fuse_active_fetal_fraction_male_and_female(self) -> None:
        meta = _make_dummy_prenatal_metadata()

        _sca_m, fused_ff, _se_m, source = _evaluate_sca_and_active_ff(
            "M", 0.10, 0.01, 0.098, 0.105, 0.95, meta, ff_size=0.102
        )
        self.assertEqual(source, "FUSED")
        self.assertAlmostEqual(fused_ff, 0.10, delta=0.01)

        _sca_f, fused_f, _se_f, source_f = _evaluate_sca_and_active_ff(
            "F", None, None, 0.0, 0.08, 1.0, meta, ff_size=0.082
        )
        self.assertEqual(source_f, "FUSED")
        self.assertAlmostEqual(fused_f, 0.081, delta=0.005)

        _sca_ffa, ffa_only, _se_ffa, source_ffa = _evaluate_sca_and_active_ff(
            "F", None, None, 0.0, 0.08, 1.0, meta, ff_size=None
        )
        self.assertEqual(source_ffa, "FFA")
        self.assertAlmostEqual(ffa_only, 0.08, places=4)

    def test_classify_segment_origin_5state_and_annotate(self) -> None:
        p_fet, p_mat = _fetal_maternal_densities()
        idx = np.asarray(FRAG_BAND_EDGES, dtype=np.int64) - FRAG_MIN_LEN
        p_f24 = np.add.reduceat(p_fet, idx[:-1])
        p_m24 = np.add.reduceat(p_mat, idx[:-1])
        p_f24 /= np.sum(p_f24)
        p_m24 /= np.sum(p_m24)
        p0_24 = 0.10 * p_f24 + 0.90 * p_m24

        phi_fc = (1.5 * 0.10) / (0.90 + 1.5 * 0.10)
        k_fc = np.round(5000.0 * (phi_fc * p_f24 + (1.0 - phi_fc) * p_m24))
        call_fc, post_fc, _, _ = classify_segment_origin_5state(
            1.05, k_fc, p_f24, p_m24, p0_24, 0.10, 0.008, True
        )
        self.assertEqual(call_fc, "FETAL_CONSTITUTIONAL")
        self.assertGreater(post_fc["FETAL_CONSTITUTIONAL"], 0.80)

        phi_mu = 0.10 / (1.5 * 0.90 + 0.10)
        k_mu = np.round(5000.0 * (phi_mu * p_f24 + (1.0 - phi_mu) * p_m24))
        call_mu, post_mu, _, _ = classify_segment_origin_5state(
            1.45, k_mu, p_f24, p_m24, p0_24, 0.10, 0.008, True
        )
        self.assertEqual(call_mu, "MATERNAL_UNINHERITED")
        self.assertGreater(post_mu["MATERNAL_UNINHERITED"], 0.80)

        empty_tbl = make_sample_table([10], starts=[0], ends=[5000], bin_size=5000)
        seg = SegmentResult("chr1", 0, 5000, math.log2(1.05), 4.0, 10.0, 1, "gain")
        annotated, _lod = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=BandPrefixSums.from_sample_table(empty_tbl),
            p_fetal_density=p_fet,
            p_maternal_density=p_mat,
            active_ff=0.10,
        )
        self.assertIsNone(annotated[0].frag_short_ratio)
        self.assertIsNone(annotated[0].frag_llr_size)
        self.assertIsNone(annotated[0].origin_call)

    def test_fit_and_predict_prenatal_with_fragmentomics(self) -> None:
        contigs = ["chr1", "chr2", "chrX", "chrY"]
        tables: list[pa.Table] = []
        entries: list[SamplesheetEntry] = []
        for i in range(6):
            is_male = i >= 3
            ff_val = 0.10 if is_male else 0.08
            frag_meta = _make_mock_fragmentomics_metadata(
                fetal_fraction=ff_val, is_male=is_male, seed=42 + i
            )
            tables.append(
                _make_mock_sample_table(
                    contigs,
                    20,
                    5000,
                    is_male,
                    frag_meta,
                    fetal_fraction=ff_val,
                )
            )
            entries.append(
                SamplesheetEntry(
                    Path(f"sample_{i}.ccnv"), "M" if is_male else "F", ff_val
                )
            )

        meta, (auto_fit, ffy_fit), _sexes = fit_prenatal_models(
            tables, entries, prenatal_bin_size=100000, seed=42
        )

        if meta.p_fetal_density is None or meta.p_maternal_density is None:
            self.fail("expected deconvolved fetal and maternal densities")
        self.assertEqual(len(meta.p_fetal_density), FRAG_NUM_BINS)
        self.assertEqual(len(meta.p_maternal_density), FRAG_NUM_BINS)
        self.assertAlmostEqual(sum(meta.p_fetal_density), 1.0, delta=1e-3)
        self.assertAlmostEqual(sum(meta.p_maternal_density), 1.0, delta=1e-3)

        if meta.ff_frag_weights is None:
            self.fail("expected ff_frag_weights")
        self.assertEqual(len(meta.ff_frag_weights), 36)
        self.assertIsNotNone(meta.ff_frag_bias)

        ref_table = assemble_reference_table([auto_fit, ffy_fit])
        res_male = predict_prenatal(
            SampleBuffers.from_table(tables[4]), ref_table, meta
        )
        self.assertEqual(res_male.fetal_sex, "M")
        self.assertIsNotNone(res_male.ff_size)
        self.assertEqual(res_male.active_ff_source, "FUSED")

        res_female = predict_prenatal(
            SampleBuffers.from_table(tables[1]), ref_table, meta
        )
        self.assertEqual(res_female.fetal_sex, "F")
        self.assertIsNotNone(res_female.ff_size)
        self.assertIn(res_female.active_ff_source, ("FUSED", "FFA"))

    def test_apply_prenatal_fisher_weighting_replaces_counts(self) -> None:
        sample_tbl = make_sample_table([10, 20], fragment_lengths=[[140, 166], [166]])
        sample_bufs = SampleBuffers.from_table(sample_tbl)
        out_noop = apply_prenatal_fisher_weighting(sample_bufs, None)
        self.assertIsInstance(out_noop, SampleBuffers)
        np.testing.assert_allclose(out_noop.count, [10.0, 20.0])

        meta = _fisher_prenatal_meta()
        out_weighted = apply_prenatal_fisher_weighting(sample_bufs, meta)
        weighted_counts = out_weighted.count
        self.assertEqual(len(weighted_counts), 2)
        self.assertGreater(float(weighted_counts[0]), 0.0)
        self.assertGreater(float(weighted_counts[1]), 0.0)

    def test_prenatal_prediction_fisher_weighting_reference_scaling(self) -> None:
        self.assertFalse(hasattr(ff_mod, "align_reference_to_weighted_sample"))

        sample_tbl = make_sample_table(
            [50, 50, 50, 50, 50],
            contigs=["chr1", "chr1", "chr2", "chr2", "chr3"],
            starts=[0, 100, 0, 100, 0],
            ends=[100, 200, 100, 200, 100],
            contig_lengths={"chr1": 200, "chr2": 200, "chr3": 100},
            bin_size=100,
            fragment_lengths=[[166] * 50] * 4 + [[135] * 50],
        )
        meta = _fisher_prenatal_meta()
        weighted_sample = apply_prenatal_fisher_weighting(
            SampleBuffers.from_table(sample_tbl), meta
        )
        if weighted_sample.kappa_frag is None:
            self.fail("expected kappa_frag")

        w_counts = np.asarray(weighted_sample.count, dtype=np.float64)
        kappa = weighted_sample.kappa_frag.astype(np.float32)
        self.assertTrue(np.all(kappa > 0.0))
        np.testing.assert_allclose(kappa[:4], w_counts[:4] / 50.0, rtol=1e-5)
        np.testing.assert_allclose(kappa[4], w_counts[4] / 50.0, rtol=1e-5)

        pw = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)).astype(np.float16)
        branch_model = BranchModel(
            "autosomal",
            ("chr1", "chr1", "chr2", "chr2", "chr3"),
            np.array([0, 100, 0, 100, 0], dtype=np.int64),
            np.array([100, 200, 100, 200, 100], dtype=np.int64),
            w_counts,
            np.zeros(5, dtype=bool),
            np.full(5, float(np.mean(w_counts[:4])), dtype=np.float64),
            np.empty((0, 5), dtype=np.float32),
            np.zeros((5, 4), dtype=np.uint16),
            pw,
            np.ones(5, dtype=np.float32),
            np.zeros(5, dtype=np.float64),
            np.full(5, 0.002, dtype=np.float64),
            bin_size=100,
            kappa_frag=kappa,
        )
        norm_res = normalize_sample_branch(
            w_counts,
            np.ones(5, dtype=np.float32),
            branch_model,
            kappa_frag=kappa,
            min_reference_bins=1,
        )
        np.testing.assert_allclose(norm_res.ratios[:4], 0.0, atol=0.05)
        self.assertGreater(float(norm_res.ratios[4]), 0.5)

        se_sample = make_sample_table([10, 20], starts=[0, 100], ends=[100, 200])
        se_bufs = SampleBuffers.from_table(se_sample)
        se_weighted = apply_prenatal_fisher_weighting(se_bufs, meta)
        self.assertIs(se_weighted, se_bufs)

    def test_defect1_prenatal_reference_and_sample_fisher_weighting_parity(
        self,
    ) -> None:
        contig_lengths = {k: 200_000 for k in ("chr1", "chr2", "chrX", "chrY")}
        bin_size, n_bins = 5000, 160
        fetal_pdf, maternal_pdf = _fetal_maternal_densities()
        rng = np.random.default_rng(101)
        bin_short_bias = np.tile([0.45, 0.08], n_bins // 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_paths: list[Path] = []
            entries: list[SamplesheetEntry] = []

            for i in range(6):
                is_male = i >= 3
                ff = 0.06 + 0.04 * (i - 3) if is_male else 0.10
                counts = np.full(n_bins, 200, dtype=np.uint32)
                counts[120:] = round(200 * 0.5 * ff) if is_male else 1
                counts[80:120] = round(200 * (1.0 - 0.5 * ff)) if is_male else 200

                bands = np.zeros((n_bins, NUM_FRAG_BANDS), dtype=np.uint32)
                c_short = np.rint(counts * bin_short_bias).astype(np.uint32)
                bands[:, 4] = c_short
                bands[:, 10] = np.maximum(0, counts.astype(np.int64) - c_short).astype(
                    np.uint32
                )

                mix_pdf = ff * fetal_pdf + (1.0 - ff) * maternal_pdf
                hist_auto = rng.multinomial(50_000, mix_pdf)
                hist_chry = (
                    rng.multinomial(2_000, fetal_pdf)
                    if is_male
                    else rng.multinomial(20, maternal_pdf)
                )
                frag_meta = _frag_meta(
                    hist_auto.tolist(),
                    hist_chry.tolist(),
                    (hist_auto // 20).tolist(),
                    [100] * 256,
                )
                sp = root / f"s_{i}.ccnv"
                bufs = _make_buffers(
                    contig_lengths, bin_size, counts, bands, frag_meta=frag_meta
                )
                write_sample_parquet(bufs.to_table(), sp)
                sample_paths.append(sp)
                entries.append(
                    SamplesheetEntry(
                        sp, "M" if is_male else "F", ff if is_male else None
                    )
                )

            sheet = root / "sheet.yaml"
            rows = [
                {"filepath": str(e.filepath), "sex": e.sex, "ff": e.ff} for e in entries
            ]
            sheet.write_text(yaml.safe_dump(rows), encoding="utf-8")
            ref_prefix = root / "prenatal_ref"
            run_reference(
                ReferenceConfig(
                    samplesheet_path=sheet,
                    prefix=ref_prefix,
                    mode="prenatal",
                    bin_size=bin_size,
                    prenatal_bin_size=100_000,
                    min_cohort_size=4,
                    min_sex_group_size=2,
                    pca_components=2,
                    qc_report=False,
                )
            )
            ref_path = parquet_path(ref_prefix)
            p_meta = PrenatalModelMetadata.from_table(pq.read_table(ref_path))
            self.assertIsNotNone(p_meta)
            if p_meta is None:
                self.fail("expected non-None PrenatalModelMetadata")
            self.assertIsNotNone(p_meta.band_fisher_weights)
            if p_meta.band_fisher_weights is None:
                self.fail("expected non-None band_fisher_weights")

            test_bufs = SampleBuffers.from_table(pq.read_table(sample_paths[4]))
            weighted_bufs = apply_prenatal_fisher_weighting(test_bufs, p_meta)
            base_cohort = load_and_assemble_cohort(entries, bin_size)
            weighted_cohort = build_fisher_weighted_cohort(
                base_cohort, p_meta.band_fisher_weights
            )
            np.testing.assert_allclose(
                weighted_cohort.counts[:, 4], weighted_bufs.count, rtol=1e-5, atol=1e-5
            )

            out_prefix = root / "pred"
            run_predict(
                PredictConfig(
                    sample_paths[4],
                    ref_path,
                    out_prefix,
                    mode="prenatal",
                    plot=False,
                    min_reference_bins=1,
                )
            )
            with gzip.open(Path(f"{out_prefix}_bins.bed.gz"), "rt") as fh:
                auto_ratios = [
                    float(line.split("\t")[4])
                    for line in fh
                    if line.startswith(("chr1\t", "chr2\t"))
                    and line.split("\t")[4] != "nan"
                ]
            self.assertGreater(len(auto_ratios), 0)
            self.assertLess(abs(float(np.median(auto_ratios))), 0.05)

    def test_unlabeled_ff_size_sizing_shift_parity_and_sca_posteriors(self) -> None:
        cl = {"chr1": 200_000, "chr2": 200_000, "chrX": 200_000, "chrY": 300_000}
        m_ffs = (0.05, 0.07, 0.09, 0.12, 0.15, 0.18)
        tables = [
            _make_unlabeled_prenatal_sample(
                cl, i >= 4, m_ffs[i - 4] if i >= 4 else 0.06 + 0.03 * i, 300 + i
            )
            for i in range(10)
        ]
        entries = [
            SamplesheetEntry(Path(f"u_{i}.ccnv"), sex="M" if i >= 4 else "F", ff=None)
            for i in range(10)
        ]
        meta, _, _ = fit_prenatal_models(tables, entries, prenatal_bin_size=100_000)
        if meta.ff_frag_weights is None:
            self.fail("Expected unlabeled paired-end cohort to train ff_frag_weights")
        self.assertEqual(len(meta.ff_frag_weights), 36)

        bp = np.arange(FRAG_MIN_LEN, FRAG_MIN_LEN + FRAG_NUM_BINS, dtype=np.float64)
        raw_hist = np.exp(-0.5 * ((bp - 166.0) / 10.0) ** 2) * 10_000.0
        shifted_hist = sf.align_histogram(raw_hist, -3.0)
        motifs = np.full(256, 100.0, dtype=np.float64)
        feat_can = sf.build_fragmentomic_feature_vector(raw_hist, motifs, 166.0, 0.0)
        feat_shf = sf.build_fragmentomic_feature_vector(
            shifted_hist, motifs, 166.0, 3.0
        )
        np.testing.assert_allclose(feat_shf, feat_can, rtol=1e-2, atol=1e-2)

        ens = sf.fit_ff_fragmentomics_ensemble(
            np.vstack([feat_can * (1.0 + 0.02 * k) for k in range(6)]),
            np.linspace(0.05, 0.20, 6),
        )

        def _pred_ff(hist: NDArray[np.float64], shift: float) -> float | None:
            return sf.predict_ff_fragmentomics(
                hist,
                motifs,
                sizing_shift_bp=shift,
                p_fetal_density=meta.p_fetal_density,
                p_maternal_density=meta.p_maternal_density,
                ff_frag_weights=ens.weights,
                ff_frag_bias=ens.intercept,
                ff_frag_feature_mean=ens.feat_mean,
                ff_frag_feature_scale=ens.feat_std,
            )

        ff_can = _pred_ff(raw_hist, 0.0)
        ff_shf = _pred_ff(shifted_hist, 3.0)
        if ff_can is None or ff_shf is None:
            self.fail("Expected non-None FF_size predictions")
        self.assertAlmostEqual(ff_shf, ff_can, places=3)

        disc_res = classify_sca_ray_4d(
            0.25, 0.01, 0.05, 0.01, 0.05, 0.015, ff_size=0.05
        )
        self.assertEqual(disc_res.call, "DISCORDANT_MIXED")
        self.assertAlmostEqual(sum(disc_res.posteriors.values()), 1.0, places=6)

        fetal_xo = classify_sca_ray_4d(
            -0.06, 0.008, 0.0, 0.01, 0.12, 0.015, ff_size=0.12
        )
        self.assertEqual(fetal_xo.call, "XO")
        maternal_45x = classify_sca_ray_4d(
            -0.06, 0.008, 0.0, 0.01, 0.12, 0.015, ff_size=0.12, chrx_llr_fetal=-10.0
        )
        self.assertEqual(maternal_45x.call, "DISCORDANT_MIXED")
        self.assertAlmostEqual(sum(maternal_45x.posteriors.values()), 1.0, places=6)

    def test_resolve_fisher_moment_weights_and_clamped_active_ff_plasma(self) -> None:
        lengths = {"chr1": 10000}
        bands = np.ones((2, NUM_FRAG_BANDS), dtype=np.uint32)
        c24 = np.array([24.0, 24.0], dtype=np.float64)
        pe_bufs = _make_buffers(lengths, 5000, c24, bands)
        se_bufs = _make_buffers(lengths, 5000, c24, None)
        meta = _make_dummy_prenatal_metadata(
            yfrac_threshold=1e-4,
            ffy_slope=2.0,
            olm_weights=[0.0],
            olm_bias=-2.2,
            band_fisher_weights=[1.0] * NUM_FRAG_BANDS,
            mu_fetal_w=1.5,
            mu_maternal_w=0.9,
        )
        self.assertEqual(resolve_fisher_moment_weights(pe_bufs, meta), (1.5, 0.9))
        self.assertEqual(resolve_fisher_moment_weights(se_bufs, meta), (1.0, 1.0))

        eff_pe = _compute_effective_ff(0.10, pe_bufs, meta)
        expected_eff = (0.10 * 1.5) / (0.10 * 1.5 + 0.90 * 0.9)
        self.assertAlmostEqual(eff_pe, expected_eff, places=6)
        self.assertAlmostEqual(_compute_effective_ff(0.10, se_bufs, meta), 0.10)

        seg = SegmentResult(
            "chr1", 0, 10000, math.log2(1.0 + 0.5 * expected_eff), 5.0, 100.0, 2, "gain"
        )
        ff_vals = ("F", 0.10, "FFA", None, 0.10, 0.10, 1.0, 0.0, 1e-4)
        ff_res = FetalFractionResult(*ff_vals, SCAResult("XX", False, 0.99, 0.0))
        ann_segs, _ = annotate_sample_prenatal_segments(
            (seg,), pe_bufs, prenatal_meta=meta, prenatal_result=ff_res
        )
        self.assertAlmostEqual(
            float(ann_segs[0].expected_ratio or 0.0),
            math.log2(1.0 + 0.5 * expected_eff),
            places=6,
        )

        ps = BandPrefixSums.from_sample_table(pe_bufs)
        ann_clamped, _ = annotate_prenatal_origin_and_lod(
            (seg,),
            prefix_sums=ps,
            sample_bands=bands,
            active_ff=1.5,
            active_ff_se=0.01,
            sca_call="XX",
        )
        self.assertIsNotNone(ann_clamped[0].origin_call)


if __name__ == "__main__":
    unittest.main()
