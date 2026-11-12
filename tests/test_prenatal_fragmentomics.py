"""Unit tests for prenatal fragmentomics cohort modeling, FF_size fusion, and segment LLR."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import asdict

import numpy as np
import pyarrow as pa
from helpers import make_sample_table
from scipy.stats import norm

from ccnv._core import FRAG_BAND_EDGES
from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    extract_sample_fragmentomics,
)
from ccnv.features.prenatal import (
    _evaluate_sca_and_active_ff,
    fit_prenatal_models,
    predict_prenatal,
)
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    annotate_prenatal_origin_and_lod,
    classify_segment_origin_5state,
)
from ccnv.models import (
    NUM_FRAG_BANDS,
    SampleBuffers,
    SamplesheetEntry,
    SegmentResult,
    assemble_reference_table,
)
from ccnv.models.schemas import FragmentomicsMetadata, PrenatalModelMetadata
from ccnv.stats.fragmentomics import (
    FRAG_MAX_LEN,
    FRAG_MIN_LEN,
    FRAG_NUM_BINS,
)


def _make_dummy_prenatal_metadata(
    **overrides: object,
) -> PrenatalModelMetadata:
    defaults: dict[str, object] = {
        "prenatal_bin_size": 100000,
        "yfrac_threshold": 0.001,
        "ffy_intercept": 0.0,
        "ffy_slope": 1.0,
        "olm_weights": [],
        "olm_bias": 0.0,
        "ffx_intercept": 2.0,
        "ffx_slope": -2.0,
        "female_chrx_ratio_mean": 1.0,
        "female_chrx_ratio_std": 0.01,
        "ffy_std": 0.01,
        "ffa_std": 0.02,
        "ff_size_std": 0.018,
    }
    defaults.update(overrides)
    return PrenatalModelMetadata.from_dict(defaults)


def _make_mock_fragmentomics_metadata(
    *,
    paired_end: bool = True,
    shift_bp: float = 0.0,
    fetal_fraction: float = 0.10,
    is_male: bool = True,
    seed: int = 42,
) -> FragmentomicsMetadata:
    rng = np.random.default_rng(seed)
    grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)

    # Maternal density peaked at 166 bp, Fetal density peaked at 143 bp
    p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
    p_fet = norm.pdf(grid, loc=143.0, scale=10.0)

    p_mix = (1.0 - fetal_fraction) * p_mat + fetal_fraction * p_fet
    hist_auto = rng.multinomial(50000, p_mix / np.sum(p_mix)).tolist()

    if is_male:
        hist_chry = rng.multinomial(1000, p_fet / np.sum(p_fet)).tolist()
    else:
        hist_chry = rng.multinomial(20, p_mat / np.sum(p_mat)).tolist()

    hist_chrx = rng.multinomial(2000, p_mix / np.sum(p_mix)).tolist()
    end_motifs = rng.poisson(100, size=256).tolist()

    return FragmentomicsMetadata(
        paired_end=paired_end,
        total_paired_fragments=50000,
        chromatosome_mode_bp=166.0,
        core_shoulder_bp=143.0,
        sizing_shift_bp=shift_bp,
        self_referenced_peak_ratio=float(
            p_mix[143 - FRAG_MIN_LEN] / p_mix[166 - FRAG_MIN_LEN]
        ),
        dinucleosome_fraction=0.08,
        high_mw_fraction=0.03,
        helical_10bp_amplitude=0.05,
        helical_10bp_phase=1.2,
        motif_diversity_score=0.92,
        hist_autosomal=hist_auto,
        hist_chry=hist_chry,
        hist_chrx=hist_chrx,
        end_motif_counts=end_motifs,
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
    c_list: list[str] = []
    s_list: list[int] = []
    e_list: list[int] = []
    counts: list[float] = []
    frag_lengths_list: list[list[int]] = []
    ep_counts: list[int] = []
    frag_gc_sums: list[float] = []

    rng = np.random.default_rng(123)

    bin_idx = 0
    for contig in contigs:
        for b in range(n_bins_per_contig):
            c_list.append(contig)
            s_list.append(b * bin_size)
            e_list.append((b + 1) * bin_size)
            if contig.endswith("Y"):
                c_val = 100.0 * (0.5 * fetal_fraction) if is_male else 0.0
            elif contig.endswith("X"):
                c_val = 100.0 * (1.0 - 0.5 * fetal_fraction) if is_male else 100.0
            else:
                c_val = 100.0
            counts.append(c_val)
            ep_counts.append(50)
            frag_gc_sums.append(42.0)

            if fetal_enriched_bins and bin_idx in fetal_enriched_bins:
                lens = rng.choice(np.arange(120, 150), size=100).tolist()
            else:
                lens = rng.choice(np.arange(155, 175), size=100).tolist()
            frag_lengths_list.append(lens)
            bin_idx += 1

    table = make_sample_table(
        counts=counts,
        contigs=c_list,
        starts=s_list,
        ends=e_list,
        contig_lengths={c: n_bins_per_contig * bin_size for c in contigs},
        bin_size=bin_size,
        fragment_lengths=frag_lengths_list,
        endpoint_counts=ep_counts,
        frag_gc_sums=frag_gc_sums,
    )
    if frag_meta is not None:
        meta_dict = dict(table.schema.metadata or {})
        meta_dict[b"ccnv.fragmentomics"] = json.dumps(asdict(frag_meta)).encode()
        table = table.replace_schema_metadata(meta_dict)
    return table


class TestPrenatalFragmentomics(unittest.TestCase):
    def test_extract_sample_fragmentomics_and_metadata(self) -> None:
        frag_meta = _make_mock_fragmentomics_metadata()
        table = _make_mock_sample_table(["chr1", "chrY"], frag_meta=frag_meta)

        extracted = extract_sample_fragmentomics(table.schema.metadata)
        self.assertIsNotNone(extracted)
        self.assertTrue(extracted.paired_end)
        self.assertEqual(len(extracted.hist_autosomal), FRAG_NUM_BINS)

    def test_band_prefix_sums_query(self) -> None:
        table = _make_mock_sample_table(
            ["chr1", "chr2"], n_bins_per_contig=5, bin_size=5000
        )
        ps = BandPrefixSums.from_sample_table(table)
        self.assertIsNotNone(ps)
        assert ps is not None

        # Bins 0-2 of chr1 -> [0, 15000) -> 3 bins * 100 fragments = 300
        bands = ps.query("chr1", 0, 15000)
        self.assertEqual(bands.shape, (NUM_FRAG_BANDS,))
        self.assertAlmostEqual(float(np.sum(bands)), 300.0)

        # Non-existent contig returns zeros
        empty = ps.query("chrUn", 0, 5000)
        self.assertAlmostEqual(float(np.sum(empty)), 0.0)

    def test_fuse_active_fetal_fraction_male_and_female(self) -> None:
        meta = _make_dummy_prenatal_metadata()

        # Male: FFY=0.10, FFA=0.105, FFX=0.098, FF_size=0.102 -> FUSED
        _sca_m, fused_ff, _se_m, source = _evaluate_sca_and_active_ff(
            "M", 0.10, 0.098, 0.105, 0.95, meta, ff_size=0.102
        )
        self.assertEqual(source, "FUSED")
        self.assertAlmostEqual(fused_ff, 0.10, delta=0.01)

        # Female: FFA=0.08, FF_size=0.082 -> FUSED
        _sca_f, fused_f, _se_f, source_f = _evaluate_sca_and_active_ff(
            "F", None, 0.0, 0.08, 1.0, meta, ff_size=0.082
        )
        self.assertEqual(source_f, "FUSED")
        self.assertAlmostEqual(fused_f, 0.081, delta=0.005)

        # Female: FF_size is None -> FFA
        _sca_ffa, ffa_only, _se_ffa, source_ffa = _evaluate_sca_and_active_ff(
            "F", None, 0.0, 0.08, 1.0, meta, ff_size=None
        )
        self.assertEqual(source_ffa, "FFA")
        self.assertAlmostEqual(ffa_only, 0.08, places=4)

    def test_classify_segment_origin_5state_and_annotate(self) -> None:
        grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_mat /= np.sum(p_mat)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_fet /= np.sum(p_fet)

        edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
        p_f24 = np.array(
            [
                np.sum(p_fet[edges[i] - FRAG_MIN_LEN : edges[i + 1] - FRAG_MIN_LEN])
                for i in range(NUM_FRAG_BANDS)
            ]
        )
        p_m24 = np.array(
            [
                np.sum(p_mat[edges[i] - FRAG_MIN_LEN : edges[i + 1] - FRAG_MIN_LEN])
                for i in range(NUM_FRAG_BANDS)
            ]
        )
        p_f24 /= np.sum(p_f24)
        p_m24 /= np.sum(p_m24)
        p0_24 = 0.10 * p_f24 + 0.90 * p_m24

        # Fetal constitutional gain: r_obs = 1.05, band counts follow fetal-shifted mixture
        phi_fc = (1.5 * 0.10) / (0.90 + 1.5 * 0.10)
        k_fc = np.round(5000.0 * (phi_fc * p_f24 + (1.0 - phi_fc) * p_m24))
        call_fc, post_fc, _, _ = classify_segment_origin_5state(
            r_obs=1.05,
            k_band_counts=k_fc,
            p_fetal_24=p_f24,
            p_maternal_24=p_m24,
            p_plasma_0=p0_24,
            active_ff=0.10,
            active_ff_se=0.008,
            is_gain=True,
        )
        self.assertEqual(call_fc, "FETAL_CONSTITUTIONAL")
        self.assertGreater(post_fc["FETAL_CONSTITUTIONAL"], 0.80)

        # Maternal uninherited gain: r_obs = 1 + 0.5 * 0.90 = 1.45, maternal-shifted bands
        phi_mu = 0.10 / (1.5 * 0.90 + 0.10)
        k_mu = np.round(5000.0 * (phi_mu * p_f24 + (1.0 - phi_mu) * p_m24))
        call_mu, post_mu, _, _ = classify_segment_origin_5state(
            r_obs=1.45,
            k_band_counts=k_mu,
            p_fetal_24=p_f24,
            p_maternal_24=p_m24,
            p_plasma_0=p0_24,
            active_ff=0.10,
            active_ff_se=0.008,
            is_gain=True,
        )
        self.assertEqual(call_mu, "MATERNAL_UNINHERITED")
        self.assertGreater(post_mu["MATERNAL_UNINHERITED"], 0.80)

        # Zero band counts in BandPrefixSums yields None fragmentomic fields
        empty_tbl = make_sample_table([10], starts=[0], ends=[5000], bin_size=5000)
        seg = SegmentResult("chr1", 0, 5000, math.log2(1.05), 4.0, 10.0, 1, "gain")
        annotated, _lod = annotate_prenatal_origin_and_lod(
            [seg],
            prefix_sums=BandPrefixSums.from_sample_table(empty_tbl),
            p_fetal_24=p_fet,
            p_maternal_24=p_mat,
            active_ff=0.10,
        )
        self.assertIsNone(annotated[0].frag_short_ratio)
        self.assertIsNone(annotated[0].frag_llr_size)
        self.assertIsNone(annotated[0].origin_call)

    def test_fit_and_predict_prenatal_with_fragmentomics(self) -> None:
        contigs = ["chr1", "chr2", "chrX", "chrY"]
        tables = []
        entries = []
        for i in range(6):
            is_male = i >= 3
            ff_val = 0.10 if is_male else 0.08
            frag_meta = _make_mock_fragmentomics_metadata(
                paired_end=True,
                shift_bp=0.0,
                fetal_fraction=ff_val,
                is_male=is_male,
                seed=42 + i,
            )
            # 20 bins * 5000 bp = 100 kb per contig
            table = _make_mock_sample_table(
                contigs,
                n_bins_per_contig=20,
                bin_size=5000,
                is_male=is_male,
                frag_meta=frag_meta,
                fetal_fraction=ff_val,
            )
            tables.append(table)
            entries.append(
                SamplesheetEntry(
                    filepath=f"sample_{i}.ccnv",
                    sex="M" if is_male else "F",
                    ff=ff_val,
                )
            )

        auto_fit, ffy_fit, meta, _sexes = fit_prenatal_models(
            tables,
            entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )

        self.assertIsNotNone(meta.p_fetal_density)
        self.assertIsNotNone(meta.p_maternal_density)
        assert meta.p_fetal_density is not None
        assert meta.p_maternal_density is not None
        self.assertEqual(len(meta.p_fetal_density), FRAG_NUM_BINS)
        self.assertEqual(len(meta.p_maternal_density), FRAG_NUM_BINS)
        self.assertAlmostEqual(sum(meta.p_fetal_density), 1.0, delta=1e-3)
        self.assertAlmostEqual(sum(meta.p_maternal_density), 1.0, delta=1e-3)

        self.assertIsNotNone(meta.ff_frag_weights)
        assert meta.ff_frag_weights is not None
        self.assertEqual(len(meta.ff_frag_weights), 36)
        self.assertIsNotNone(meta.ff_frag_bias)

        # Predict on male sample (sample 4)
        ref_table = assemble_reference_table([auto_fit, ffy_fit])
        res_male = predict_prenatal(tables[4], ref_table, meta)
        self.assertEqual(res_male.fetal_sex, "M")
        self.assertIsNotNone(res_male.ff_size)
        self.assertEqual(res_male.active_ff_source, "FUSED")

        # Predict on female sample (sample 1)
        res_female = predict_prenatal(tables[1], ref_table, meta)
        self.assertEqual(res_female.fetal_sex, "F")
        self.assertIsNotNone(res_female.ff_size)
        self.assertIn(res_female.active_ff_source, ("FUSED", "FFA"))

    def test_apply_prenatal_fisher_weighting_replaces_counts(self) -> None:
        # 1. No-op when prenatal_meta is None
        sample_tbl = make_sample_table([10, 20], fragment_lengths=[[140, 166], [166]])
        out_noop = apply_prenatal_fisher_weighting(sample_tbl, None)
        self.assertIsInstance(out_noop, SampleBuffers)
        np.testing.assert_allclose(out_noop.count, [10.0, 20.0])

        # 2. Replaces counts when densities provided
        grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_mat /= np.sum(p_mat)
        p_fet /= np.sum(p_fet)

        meta = _make_dummy_prenatal_metadata(
            p_fetal_density=p_fet.tolist(),
            p_maternal_density=p_mat.tolist(),
        )
        out_weighted = apply_prenatal_fisher_weighting(sample_tbl, meta)
        weighted_counts = out_weighted.count
        self.assertEqual(len(weighted_counts), 2)
        self.assertGreater(float(weighted_counts[0]), 0.0)
        self.assertGreater(float(weighted_counts[1]), 0.0)

    def test_prenatal_prediction_fisher_weighting_reference_scaling(self) -> None:
        import ccnv.features.fragmentomics as ff_mod
        from ccnv.predict import BranchModel, normalize_sample_branch

        self.assertFalse(hasattr(ff_mod, "align_reference_to_weighted_sample"))

        # 5 bins with identical unweighted counts (50 reads each) across 3 contigs:
        # Bins 0..3 are diploid background (166 bp); bin 4 is fetal-enriched (135 bp)
        sample_tbl = make_sample_table(
            [50, 50, 50, 50, 50],
            contigs=["chr1", "chr1", "chr2", "chr2", "chr3"],
            starts=[0, 100, 0, 100, 0],
            ends=[100, 200, 100, 200, 100],
            contig_lengths={"chr1": 200, "chr2": 200, "chr3": 100},
            bin_size=100,
            fragment_lengths=[[166] * 50] * 4 + [[135] * 50],
        )
        grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_mat /= np.sum(p_mat)
        p_fet /= np.sum(p_fet)
        meta = _make_dummy_prenatal_metadata(
            p_fetal_density=p_fet.tolist(),
            p_maternal_density=p_mat.tolist(),
        )
        weighted_sample = apply_prenatal_fisher_weighting(sample_tbl, meta)
        self.assertIsNotNone(weighted_sample.kappa_frag)
        assert weighted_sample.kappa_frag is not None

        w_counts = np.asarray(weighted_sample.count, dtype=np.float64)
        kappa = weighted_sample.kappa_frag
        self.assertTrue(np.all(kappa >= 1.0))

        # Reference baseline is directly in weighted space; off-contig prototype 0 pools diploid bins
        bg_mean = float(np.mean(w_counts[:4]))
        branch_model = BranchModel(
            branch="autosomal",
            contigs=("chr1", "chr1", "chr2", "chr2", "chr3"),
            starts=np.array([0, 100, 0, 100, 0], dtype=np.int64),
            ends=np.array([100, 200, 100, 200, 100], dtype=np.int64),
            counts=w_counts,
            masked=np.zeros(5, dtype=bool),
            pca_mean=np.full(5, bg_mean, dtype=np.float64),
            pca_loadings=np.empty((0, 5), dtype=np.float32),
            proto_ids=np.zeros((5, 4), dtype=np.uint16),
            proto_weights=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)).astype(np.float16),
            mean_distance=np.ones(5, dtype=np.float32),
            null_mean=np.zeros(5, dtype=np.float64),
            bio_dispersion=np.full(5, 0.002, dtype=np.float64),
            bin_size=100,
            kappa_frag=kappa.astype(np.float32),
        )
        norm_res = normalize_sample_branch(
            w_counts,
            np.ones(5, dtype=np.float32),
            branch_model,
            kappa_frag=kappa.astype(np.float32),
            min_reference_bins=1,
        )
        np.testing.assert_allclose(norm_res.ratios[:4], 0.0, atol=0.05)
        self.assertGreater(norm_res.ratios[4], 0.5)

        # Confirm unweighted fallback when fragment lengths unavailable
        se_sample = make_sample_table([10, 20], starts=[0, 100], ends=[100, 200])
        se_bufs = SampleBuffers.from_table(se_sample)
        se_weighted = apply_prenatal_fisher_weighting(se_bufs, meta)
        self.assertIs(se_weighted, se_bufs)


if __name__ == "__main__":
    unittest.main()
