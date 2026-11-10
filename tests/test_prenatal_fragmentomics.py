"""Unit tests for prenatal fragmentomics cohort modeling, FF_size fusion, and segment LLR."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict

import numpy as np
import pyarrow as pa
from helpers import make_sample_table
from scipy.stats import norm

from ccnv.features.fragmentomics import (
    apply_prenatal_fisher_weighting,
    extract_sample_fragmentomics,
    extract_segment_fragment_lengths,
    fuse_active_fetal_fraction,
    score_segment_origin,
)
from ccnv.features.prenatal import (
    fit_prenatal_models,
    predict_prenatal,
)
from ccnv.models import (
    SamplesheetEntry,
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
        "mlp_w1": [],
        "mlp_b1": [],
        "mlp_w2": [],
        "mlp_b2": [],
        "mlp_skew_intercept": 0.0,
        "mlp_skew_slope": 1.0,
        "olm_weights": [],
        "olm_bias": 0.0,
        "olm_skew_intercept": 0.0,
        "olm_skew_slope": 1.0,
        "ffx_intercept": 2.0,
        "ffx_slope": -2.0,
        "female_chrx_ratio_mean": 1.0,
        "female_chrx_ratio_std": 0.01,
        "ffy_std": 0.01,
        "ffa_std": 0.02,
        "ff_size_std": 0.018,
    }
    defaults.update(overrides)
    return PrenatalModelMetadata(**defaults)  # type: ignore[arg-type]


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
            if not is_male and contig.endswith("Y"):
                counts.append(0.0)
            else:
                counts.append(100.0)
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

    def test_extract_segment_fragment_lengths(self) -> None:
        table = _make_mock_sample_table(
            ["chr1", "chr2"], n_bins_per_contig=5, bin_size=5000
        )
        # Bins 0-2 of chr1 -> [0, 15000)
        lens = extract_segment_fragment_lengths(table, "chr1", 0, 15000)
        self.assertEqual(lens.size, 300)

        # Non-existent contig
        empty = extract_segment_fragment_lengths(table, "chrUn", 0, 5000)
        self.assertEqual(empty.size, 0)

    def test_fuse_active_fetal_fraction_male_and_female(self) -> None:
        meta = _make_dummy_prenatal_metadata()

        # Male: FFY=0.10, FFA=0.105, FFX=0.098, FF_size=0.102 -> all match tolerance -> FUSED
        fused_ff, source = fuse_active_fetal_fraction(
            "M", "XY", ffy=0.10, ffa=0.105, ffx=0.098, ff_size=0.102, prenatal_meta=meta
        )
        self.assertEqual(source, "FUSED")
        self.assertAlmostEqual(fused_ff, 0.10, delta=0.01)

        # Female: FFA=0.08, FF_size=0.082 -> match tolerance -> FUSED
        fused_f, source_f = fuse_active_fetal_fraction(
            "F", "XX", ffy=None, ffa=0.08, ffx=0.0, ff_size=0.082, prenatal_meta=meta
        )
        self.assertEqual(source_f, "FUSED")
        self.assertAlmostEqual(fused_f, 0.081, delta=0.005)

        # Female: FF_size is None -> FFA
        ffa_only, source_ffa = fuse_active_fetal_fraction(
            "F", "XX", ffy=None, ffa=0.08, ffx=0.0, ff_size=None, prenatal_meta=meta
        )
        self.assertEqual(source_ffa, "FFA")
        self.assertEqual(ffa_only, 0.08)

    def test_score_segment_origin(self) -> None:
        grid = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_mat = norm.pdf(grid, loc=166.0, scale=12.0)
        p_mat /= np.sum(p_mat)
        p_fet = norm.pdf(grid, loc=143.0, scale=10.0)
        p_fet /= np.sum(p_fet)

        # Fetal-enriched lengths (short ~135 bp)
        fetal_lens = np.full(500, 135, dtype=np.int64)
        sr, llr, origin = score_segment_origin(
            fetal_lens,
            shift_bp=0.0,
            p_fetal=p_fet,
            p_maternal=p_mat,
            active_ff=0.10,
            is_gain=True,
            has_call=True,
        )
        self.assertIsNotNone(sr)
        self.assertGreater(sr, 0.9)
        self.assertIsNotNone(llr)
        self.assertGreater(llr, 2.0)
        self.assertEqual(origin, "FETAL_CONFIRMED")

        # Maternal-enriched lengths (long ~170 bp)
        mat_lens = np.full(500, 170, dtype=np.int64)
        _sr_m, llr_m, origin_m = score_segment_origin(
            mat_lens,
            shift_bp=0.0,
            p_fetal=p_fet,
            p_maternal=p_mat,
            active_ff=0.10,
            is_gain=True,
            has_call=True,
        )
        self.assertIsNotNone(llr_m)
        self.assertLess(llr_m, -2.0)
        self.assertEqual(origin_m, "MATERNAL_CONFIRMED")

        # Unknown-only (0) fragment lengths return (None, None, None)
        self.assertEqual(
            score_segment_origin(
                np.zeros(10, dtype=np.int64),
                shift_bp=0.0,
                p_fetal=p_fet,
                p_maternal=p_mat,
                active_ff=0.10,
                is_gain=True,
                has_call=True,
            ),
            (None, None, None),
        )

    def test_fit_and_predict_prenatal_with_fragmentomics(self) -> None:
        contigs = ["chr1", "chr2", "chrX", "chrY"]
        tables = []
        entries = []
        for i in range(6):
            is_male = i >= 3
            frag_meta = _make_mock_fragmentomics_metadata(
                paired_end=True,
                shift_bp=0.0,
                fetal_fraction=0.10 if is_male else 0.08,
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
            )
            tables.append(table)
            entries.append(
                SamplesheetEntry(
                    filepath=f"sample_{i}.ccnv",
                    sex="M" if is_male else "F",
                    ff=0.10 if is_male else 0.08,
                )
            )

        auto_tbl, ffy_tbl, meta, _sexes = fit_prenatal_models(
            tables,
            entries,
            bin_size=5000,
            prenatal_bin_size=100000,
            seed=42,
        )

        self.assertIsNotNone(meta.p_fetal_density)
        self.assertIsNotNone(meta.p_maternal_density)
        self.assertEqual(len(meta.p_fetal_density), FRAG_NUM_BINS)
        self.assertEqual(len(meta.p_maternal_density), FRAG_NUM_BINS)
        self.assertAlmostEqual(sum(meta.p_fetal_density), 1.0, delta=1e-3)
        self.assertAlmostEqual(sum(meta.p_maternal_density), 1.0, delta=1e-3)

        self.assertIsNotNone(meta.ff_frag_weights)
        self.assertEqual(len(meta.ff_frag_weights), 36)
        self.assertIsNotNone(meta.ff_frag_bias)

        # Predict on male sample (sample 4)
        ref_table = pa.concat_tables([auto_tbl, ffy_tbl])
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
        self.assertEqual(out_noop.column("count").to_pylist(), [10.0, 20.0])

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
        weighted_counts = out_weighted.column("count").to_numpy()
        self.assertEqual(len(weighted_counts), 2)
        self.assertGreater(weighted_counts[0], 0.0)
        self.assertGreater(weighted_counts[1], 0.0)

    def test_prenatal_prediction_fisher_weighting_reference_scaling(self) -> None:
        from ccnv.features.fragmentomics import align_reference_to_weighted_sample
        from ccnv.predict import BranchModel, project_coverage

        # 5 bins with identical unweighted counts (10 reads each):
        # Bins 0..3 are diploid background (166 bp); bin 4 is fetal-enriched (135 bp)
        sample_tbl = make_sample_table(
            [10, 10, 10, 10, 10],
            contigs=["chr1"] * 5,
            starts=[0, 100, 200, 300, 400],
            ends=[100, 200, 300, 400, 500],
            contig_lengths={"chr1": 500},
            bin_size=100,
            fragment_lengths=[[166] * 10] * 4 + [[135] * 10],
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
        self.assertIsNot(weighted_sample, sample_tbl)

        ref_table = pa.table(
            {
                "branch": pa.array(["autosomal"] * 5),
                "contig": pa.array(["chr1"] * 5),
                "start": pa.array([0, 100, 200, 300, 400], type=pa.int64()),
                "end": pa.array([100, 200, 300, 400, 500], type=pa.int64()),
                "pca_mean": pa.array([10.0] * 5, type=pa.float64()),
                "null_mean": pa.array([0.0] * 5, type=pa.float64()),
            }
        )
        weighted_ref = align_reference_to_weighted_sample(
            ref_table, sample_tbl, weighted_sample
        )
        u_counts = sample_tbl.column("count").to_numpy()
        w_counts = weighted_sample.column("count").to_numpy()
        expected_baseline_scale = float(np.median(w_counts / u_counts))
        np.testing.assert_allclose(
            weighted_ref.column("pca_mean").to_numpy(),
            np.full(5, 10.0 * expected_baseline_scale),
        )
        np.testing.assert_allclose(
            weighted_ref.column("null_mean").to_numpy(),
            np.zeros(5),
        )

        # Verify project_coverage preserves the Fisher-weighted enrichment in bin 4
        # under mean reconstruction (rank 0 PCA loadings, unchanged unweighted counts)
        branch_model = BranchModel(
            branch="autosomal",
            contigs=("chr1",) * 5,
            starts=np.array([0, 100, 200, 300, 400], dtype=np.int64),
            ends=np.array([100, 200, 300, 400, 500], dtype=np.int64),
            counts=w_counts.astype(np.float64),
            masked=np.zeros(5, dtype=bool),
            pca_mean=weighted_ref.column("pca_mean").to_numpy().astype(np.float64),
            pca_loadings=np.empty((0, 5), dtype=np.float64),
            nearest_indices=(np.array([1], dtype=np.int32),) * 5,
            mean_distance=np.ones(5, dtype=np.float32),
            null_mean=np.zeros(5, dtype=np.float64),
            null_std=np.full(5, 0.05, dtype=np.float64),
        )
        projected = project_coverage(branch_model)
        np.testing.assert_allclose(projected.corrected[:4], 1.0, rtol=1e-6)
        self.assertGreater(projected.corrected[4], 1.5)

        # Confirm unweighted fallback when fragment lengths unavailable
        se_sample = make_sample_table([10, 20], starts=[0, 100], ends=[100, 200])
        se_weighted = apply_prenatal_fisher_weighting(se_sample, meta)
        self.assertIs(se_weighted, se_sample)


if __name__ == "__main__":
    unittest.main()
