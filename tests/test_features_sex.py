"""Unit tests for 1 Mb super-block depth scaling and 2D germline sex inference/quarantine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from helpers import make_sample_table
from numpy.typing import NDArray

from ccnv.features.binning import compute_bin_exposure
from ccnv.features.cohort import (
    Cohort,
    estimate_size_factors,
    load_and_assemble_cohort,
    mask_low_coverage_per_contig,
)
from ccnv.features.sex import (
    _sum_counts_by_sample,
    infer_germline_sex_2d,
    infer_sample_sexes,
)
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    SCHEMA_VERSION,
    FragmentomicsMetadata,
    InvalidInputError,
    PredictConfig,
    ReferenceConfig,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    decode_dictionary_contigs,
)
from ccnv.predict import load_inputs
from ccnv.predict import run as run_predict
from ccnv.reference import build_reference_table


def _simulate_human_sex_cohort(
    karyotypes: list[tuple[str, float, float, str]],
    *,
    seed: int,
    xtr_mismap_frac: float = 0.002,
    msy_mismap_frac: float = 0.0,
    par_y_factor: float = 0.0,
    depth_range: tuple[float, float] = (0.85, 1.20),
) -> tuple[Cohort, NDArray[np.bool_], tuple[SamplesheetEntry, ...]]:
    rng = np.random.default_rng(seed)
    bin_size = 500_000
    contig_lengths = {
        "chr1": 60_000_000,
        "chr2": 60_000_000,
        "chrX": 60_000_000,
        "chrY": 58_000_000,
    }
    contigs_list: list[str] = []
    starts_list: list[int] = []
    ends_list: list[int] = []
    for cname, clen in contig_lengths.items():
        for s in range(0, clen, bin_size):
            contigs_list.append(cname)
            starts_list.append(s)
            ends_list.append(min(s + bin_size, clen))

    contigs_arr = np.asarray(contigs_list, dtype=object)
    starts_arr = np.asarray(starts_list, dtype=np.int64)
    ends_arr = np.asarray(ends_list, dtype=np.int64)
    n_bins = starts_arr.size
    n_samples = len(karyotypes)

    is_auto = np.isin(contigs_arr, ["chr1", "chr2"])
    is_x = contigs_arr == "chrX"
    is_y = contigs_arr == "chrY"
    is_x_par = is_x & ((starts_arr < 2_500_000) | (ends_arr > 59_500_000))
    is_y_par = is_y & ((starts_arr < 2_500_000) | (ends_arr > 57_500_000))
    is_y_het = is_y & (starts_arr >= 28_000_000) & (ends_arr <= 57_000_000)
    is_y_xtr = is_y & (starts_arr >= 4_000_000) & (ends_arr <= 6_000_000)

    counts = np.zeros((n_bins, n_samples), dtype=np.float32)
    for s_idx, (_, ploidy_x, ploidy_y, _) in enumerate(karyotypes):
        diploid_mu = 400.0 * rng.uniform(*depth_range)
        mu = np.zeros(n_bins, dtype=np.float64)
        mu[is_auto] = diploid_mu
        mu[is_x_par] = diploid_mu
        mu[is_y_par] = diploid_mu * par_y_factor
        mu[is_x & ~is_x_par] = diploid_mu * (ploidy_x / 2.0)
        mu[is_y & ~is_y_par & ~is_y_het] = diploid_mu * (
            ploidy_y / 2.0 + msy_mismap_frac
        )
        mu[is_y_xtr] += diploid_mu * xtr_mismap_frac
        counts[:, s_idx] = rng.poisson(mu).astype(np.float32)

    cohort = Cohort(
        counts=counts,
        contigs=tuple(contigs_list),
        starts=starts_arr,
        ends=ends_arr,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
        identities=tuple({"id": f"s_{i}"} for i in range(n_samples)),
        paths=tuple(Path(f"s_{i}.ccnv") for i in range(n_samples)),
    )
    exposure = compute_bin_exposure(starts_arr, ends_arr, bin_size)
    _, contig_codes = np.unique(contigs_arr.astype(str), return_inverse=True)
    usable = mask_low_coverage_per_contig(
        counts, contig_codes.astype(np.int16), exposure
    )
    entries = tuple(
        SamplesheetEntry(filepath=Path(f"s_{i}.ccnv"), sex="M" if lbl == "M" else "F")
        for i, (_, _, _, lbl) in enumerate(karyotypes)
    )
    return cohort, usable, entries


def _cohort_sample_table(cohort: Cohort, sample_idx: int) -> pa.Table:
    return make_sample_table(
        cohort.counts[:, sample_idx].astype(np.uint32).tolist(),
        contigs=list(cohort.contigs),
        starts=cohort.starts.tolist(),
        ends=cohort.ends.tolist(),
        contig_lengths=cohort.contig_lengths,
        bin_size=cohort.bin_size,
        identity={"id": f"s_{sample_idx}"},
    )


def _make_sex_sample_buffers(
    contig_lengths: dict[str, int],
    bin_size: int,
    counts: NDArray[np.uint32],
    *,
    sample_id: str = "S1",
    with_fragmentomics: bool = False,
    fetal_like: bool = False,
) -> SampleBuffers:
    n = int(counts.size)
    frag_meta: FragmentomicsMetadata | None = None
    bands: NDArray[np.uint32] | None = None
    paired_cnt = np.zeros(n, dtype=np.uint32)
    short_cnt = np.zeros(n, dtype=np.uint32)
    frag_len_sum = np.zeros(n, dtype=np.uint64)
    ep_cnt = np.zeros(n, dtype=np.uint32)
    gc_sum = np.zeros(n, dtype=np.float32)

    if with_fragmentomics:
        paired_cnt = counts.copy()
        short_frac = 0.28 if fetal_like else 0.16
        short_cnt = np.round(counts.astype(np.float64) * short_frac).astype(np.uint32)
        frag_len_sum = counts.astype(np.uint64) * np.uint64(162 if fetal_like else 168)
        ep_cnt = counts * np.uint32(2)
        gc_sum = counts.astype(np.float32) * np.float32(0.42)
        bands = np.zeros((n, NUM_FRAG_BANDS), dtype=np.uint32)
        bands[:, 6] = short_cnt
        bands[:, 10] = counts - short_cnt

        hist_auto = [0] * 571
        hist_y = [0] * 571
        hist_x = [0] * 571
        hist_auto[113] = 300 if fetal_like else 150
        hist_auto[137] = 700 if fetal_like else 850
        hist_x[113] = hist_auto[113] // 10
        hist_x[137] = hist_auto[137] // 10
        if fetal_like:
            hist_y[113] = 40
            hist_y[137] = 20
        frag_meta = FragmentomicsMetadata(
            paired_end=True,
            total_paired_fragments=int(np.sum(counts)),
            chromatosome_mode_bp=167.0,
            core_shoulder_bp=143.0,
            sizing_shift_bp=0.0,
            self_referenced_peak_ratio=0.35 if fetal_like else 0.20,
            dinucleosome_fraction=0.05,
            high_mw_fraction=0.01,
            helical_10bp_amplitude=0.08,
            helical_10bp_phase=0.0,
            motif_diversity_score=0.98,
            hist_autosomal=hist_auto,
            hist_chry=hist_y,
            hist_chrx=hist_x,
            end_motif_counts=[10] * 256,
        )

    meta = SampleMetadata(
        schema_version=SCHEMA_VERSION,
        coordinate_system=COORDINATE_SYSTEM,
        bin_size=bin_size,
        contig_order=tuple(contig_lengths.keys()),
        contig_lengths=contig_lengths,
        source_identity={"id": sample_id},
        reference_identity={"name": "ref"},
        filters={"min_mapping_quality": 1},
        qc={
            "effective_reads_examined": int(np.sum(counts)),
            "effective_reads_kept": int(np.sum(counts)),
        },
        fragmentomics=frag_meta,
    )
    return SampleBuffers(
        meta=meta,
        count=counts,
        paired_count=paired_cnt,
        frag_len_sum=frag_len_sum,
        short_frag_count=short_cnt,
        endpoint_count=ep_cnt,
        frag_gc_sum=gc_sum,
        frag_bands=bands,
    )


class TestFeaturesSexAndSuperblocks(unittest.TestCase):
    def test_superblock_size_factors(self) -> None:
        counts = np.array([[10.0, 20.0], [30.0, 60.0]], dtype=np.float32)
        contigs = ("chr1", "chr1")
        starts = np.array([0, 100], dtype=np.int64)
        ends = np.array([100, 200], dtype=np.int64)
        usable = np.ones(2, dtype=np.bool_)
        sf = estimate_size_factors(
            counts, usable, contigs=contigs, starts=starts, ends=ends, block_bp=100
        )
        np.testing.assert_allclose(sf[1] / sf[0], 2.0, rtol=1e-6)

    def test_superblock_size_factors_zero_sum_fails(self) -> None:
        counts = np.array([[0.0, 10.0], [0.0, 20.0]], dtype=np.float32)
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                counts,
                np.ones(2, dtype=np.bool_),
                contigs=("chr1", "chr1"),
                starts=np.array([0, 100], dtype=np.int64),
                ends=np.array([100, 200], dtype=np.int64),
                block_bp=100,
            )

    def test_infer_sample_sexes(self) -> None:
        fractions = np.array([0.001, 0.0012, 0.0009, 0.02, 0.021, 0.019])
        result = infer_sample_sexes(fractions, min_group_size=2)
        self.assertTrue(result.passed)
        self.assertEqual(
            result.labels, ("female", "female", "female", "male", "male", "male")
        )
        self.assertIsNotNone(result.cutoff)

    def test_infer_sample_sexes_cutoff(self) -> None:
        fractions = np.array([0.005, 0.025])
        result = infer_sample_sexes(fractions, min_group_size=1, cutoff=0.01)
        self.assertTrue(result.passed)
        self.assertEqual(result.labels, ("female", "male"))


class TestSuperblockDepthAndSex(unittest.TestCase):
    def test_superblock_size_factors_unquantized_at_01x_1kb_s200(self) -> None:
        rng = np.random.default_rng(42)
        bin_size = 1_000
        n_contigs = 12
        bins_per_contig = 12_000
        n_bins = n_contigs * bins_per_contig
        n_samples = 200

        contig_ids = np.repeat(np.arange(n_contigs, dtype=np.int16), bins_per_contig)
        local_starts = np.arange(bins_per_contig, dtype=np.uint32) * np.uint32(bin_size)
        local_ends = local_starts + np.uint32(bin_size)
        local_ends[-1] = local_starts[-1] + np.uint32(450)
        starts = np.tile(local_starts, n_contigs)
        ends = np.tile(local_ends, n_contigs)
        exposure = compute_bin_exposure(starts, ends, bin_size)

        usable = np.ones(n_bins, dtype=np.bool_)
        for c in range(n_contigs):
            base = c * bins_per_contig
            usable[base + 5500 : base + 6500] = False

        true_depths = np.linspace(0.10, 0.35, n_samples, dtype=np.float64)
        lambdas = (true_depths / 0.30) * 1.0
        rate_matrix = (
            exposure[:, np.newaxis].astype(np.float64) * lambdas[np.newaxis, :]
        )
        rate_matrix[:5_000, :10] *= 4.0
        counts = rng.poisson(rate_matrix).astype(np.float32)
        counts[~usable, :] = 0.0

        sf = estimate_size_factors(
            counts, usable, contigs=contig_ids, starts=starts, ends=ends
        )
        self.assertEqual(sf.dtype, np.float64)
        self.assertAlmostEqual(float(np.median(sf)), 1.0, places=12)
        self.assertEqual(np.unique(np.round(sf, 6)).size, n_samples)

        rel_err = np.abs(sf / np.median(sf) - true_depths / np.median(true_depths)) / (
            true_depths / np.median(true_depths)
        )
        self.assertLess(float(np.max(rel_err)), 0.04)

        sf_sub = estimate_size_factors(
            counts[:, :8], usable, contigs=contig_ids, starts=starts, ends=ends
        )
        normed = counts[:, :8] / sf_sub[np.newaxis, :]
        diploid_usable = usable & (contig_ids > 0)
        self.assertGreater(float(np.mean(normed[diploid_usable, 0])), 0.25)
        self.assertLess(float(np.mean(normed[diploid_usable, 0])), 0.50)

        zero_col = counts[:, :4].copy()
        zero_col[:, 2] = 0.0
        with self.assertRaises(InvalidInputError):
            estimate_size_factors(
                zero_col, usable, contigs=contig_ids, starts=starts, ends=ends
            )

    def test_per_contig_coverage_mask_preserves_clean_prenatal_chry_at_8pct_ff(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        bin_size = 5_000
        n_auto = 400
        n_chry = 200
        n_bins = n_auto + n_chry
        n_samples = 24

        contig_ids = np.concatenate(
            [np.zeros(n_auto, dtype=np.int16), np.ones(n_chry, dtype=np.int16)]
        )
        starts = np.concatenate(
            [
                np.arange(n_auto, dtype=np.uint32) * np.uint32(bin_size),
                np.arange(n_chry, dtype=np.uint32) * np.uint32(bin_size),
            ]
        )
        ends = starts + np.uint32(bin_size)
        ends[-1] = starts[-1] + np.uint32(1_000)
        exposure = compute_bin_exposure(starts, ends, bin_size)

        counts_prenatal_8pct_ff = np.zeros((n_bins, n_samples), dtype=np.float32)
        auto_lambda = 200.0
        counts_prenatal_8pct_ff[:n_auto, :] = rng.poisson(
            auto_lambda, size=(n_auto, n_samples)
        ).astype(np.float32)

        chry_clean_local = np.zeros(n_chry, dtype=np.bool_)
        chry_clean_local[10:120] = True
        chry_clean_local[-1] = True
        chry_clean_mask = np.zeros(n_bins, dtype=np.bool_)
        chry_clean_mask[n_auto:n_bins] = chry_clean_local

        for idx in np.flatnonzero(chry_clean_mask):
            counts_prenatal_8pct_ff[idx, 12:] = rng.poisson(
                0.04 * auto_lambda * float(exposure[idx]), size=12
            ).astype(np.float32)

        chry_dead_telomere_idx = n_auto + 0
        counts_prenatal_8pct_ff[chry_dead_telomere_idx, :] = 0.0
        chry_mismap_idx = n_auto + 150
        counts_prenatal_8pct_ff[chry_mismap_idx, 0] = 1.0

        keep = mask_low_coverage_per_contig(
            counts_prenatal_8pct_ff, contig_ids, exposure, min_fraction=0.05
        )
        self.assertGreater(float(np.mean(keep[chry_clean_mask])), 0.95)
        self.assertFalse(bool(keep[chry_dead_telomere_idx]))
        self.assertFalse(bool(keep[chry_mismap_idx]))
        self.assertTrue(bool(keep[-1]))

    def test_2d_sex_quarantines_xxy_and_mloy_and_persists_cutoff_when_labeled(
        self,
    ) -> None:
        karyotypes = (
            [("XX", 2.0, 0.0, "F")] * 10
            + [("XY", 1.0, 1.0, "M")] * 10
            + [
                ("XXY", 2.0, 1.0, "M"),
                ("mLOY", 1.0, 0.35, "M"),
                ("XO", 1.0, 0.0, "F"),
                ("XYY", 1.0, 2.0, "M"),
            ]
        )
        xxy_idx, mloy_idx, xo_idx, xyy_idx = 20, 21, 22, 23
        cohort_with_xxy_and_mloy, usable, labeled_entries = _simulate_human_sex_cohort(
            karyotypes,
            seed=123,
            xtr_mismap_frac=0.0,
            msy_mismap_frac=0.002,
            par_y_factor=0.5,
            depth_range=(0.8, 1.25),
        )

        sex_inf = infer_germline_sex_2d(
            cohort_with_xxy_and_mloy, usable, min_group_size=5
        )
        self.assertTrue(sex_inf.passed)
        self.assertTrue(bool(sex_inf.quarantined_mask[xxy_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[mloy_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[xo_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[xyy_idx]))
        self.assertFalse(bool(np.any(sex_inf.quarantined_mask[:20])))
        self.assertIsNotNone(sex_inf.cutoff)
        if sex_inf.cutoff is None:
            self.fail("expected non-None sex_inf.cutoff")
        self.assertGreater(sex_inf.cutoff, 0.0)
        self.assertIsNotNone(sex_inf.labels)

        ref_unlabeled = build_reference_table(
            cohort_with_xxy_and_mloy,
            ReferenceConfig(
                entries=tuple(
                    SamplesheetEntry(filepath=e.filepath) for e in labeled_entries
                ),
                prefix=Path("ref_unlabeled"),
                pca_components=2,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        ref_labeled = build_reference_table(
            cohort_with_xxy_and_mloy,
            ReferenceConfig(
                entries=labeled_entries,
                prefix=Path("ref_labeled"),
                pca_components=2,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )

        meta_labeled = ReferenceMetadata.from_table(ref_labeled)
        self.assertEqual(meta_labeled.sex_source, "samplesheet")
        self.assertIsNotNone(meta_labeled.sex_cutoff)
        if meta_labeled.sex_cutoff is None:
            self.fail("expected non-None meta_labeled.sex_cutoff")
        self.assertGreater(meta_labeled.sex_cutoff, 0.0)
        self.assertIsNotNone(meta_labeled.sex_inference)
        if meta_labeled.sex_inference is None:
            self.fail("expected non-None meta_labeled.sex_inference")
        self.assertIsNotNone(meta_labeled.sex_inference.get("cutoff"))

        for tbl in (ref_unlabeled, ref_labeled):
            b_col = np.asarray(tbl.column("branch").to_pylist(), dtype=object)
            c_col = np.asarray(tbl.column("contig").to_pylist(), dtype=object)
            s_col = tbl.column("start").to_numpy()
            e_col = tbl.column("end").to_numpy()
            m_col = tbl.column("masked").to_numpy()
            mu_col = tbl.column("pca_mean").to_numpy()

            male_x = (
                (b_col == BRANCH_MALE_SEX)
                & (c_col == "chrX")
                & ~m_col
                & (s_col >= 3_000_000)
                & (e_col <= 59_000_000)
            )
            male_y = (
                (b_col == BRANCH_MALE_SEX)
                & (c_col == "chrY")
                & ~m_col
                & (s_col >= 3_000_000)
                & (e_col <= 27_000_000)
            )
            self.assertAlmostEqual(
                float(np.median(mu_col[male_y]) / np.median(mu_col[male_x])),
                1.0,
                delta=0.05,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_path = root / "ref_labeled.ccnv"
            xx_path = root / "xx_sample.ccnv"
            xy_path = root / "xy_sample.ccnv"
            write_reference_parquet(ref_labeled, ref_path)
            write_sample_parquet(
                _cohort_sample_table(cohort_with_xxy_and_mloy, 0), xx_path
            )
            write_sample_parquet(
                _cohort_sample_table(cohort_with_xxy_and_mloy, 10), xy_path
            )

            in_xx = load_inputs(xx_path, ref_path, sex="autosomal", mode="germline")
            in_xy = load_inputs(xy_path, ref_path, sex="autosomal", mode="germline")
            self.assertEqual(in_xx.sex, "female")
            self.assertEqual(in_xx.branches, (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX))
            self.assertEqual(in_xy.sex, "male")
            self.assertEqual(in_xy.branches, (BRANCH_AUTOSOMAL, BRANCH_MALE_SEX))

    def test_single_sex_all_male_labeled_quarantines_xxy_and_mloy(self) -> None:
        karyotypes = [("XY", 1.0, 1.0, "M")] * 18 + [
            ("XXY", 2.0, 1.0, "M"),
            ("mLOY", 1.0, 0.35, "M"),
        ]
        cohort, usable, entries = _simulate_human_sex_cohort(karyotypes, seed=201)
        sex_inf = infer_germline_sex_2d(
            cohort,
            usable,
            min_group_size=5,
            explicit_labels=tuple("male" for _ in entries),
        )
        self.assertTrue(sex_inf.passed)
        self.assertFalse(bool(np.any(sex_inf.quarantined_mask[:18])))
        self.assertTrue(bool(sex_inf.quarantined_mask[18]))
        self.assertTrue(bool(sex_inf.quarantined_mask[19]))
        self.assertIsNotNone(sex_inf.cutoff)
        if sex_inf.cutoff is None:
            self.fail("expected non-None sex_inf.cutoff")
        self.assertLess(
            sex_inf.cutoff, float(np.min(np.asarray(sex_inf.fractions)[:18]))
        )

        sex_cut = infer_germline_sex_2d(cohort, usable, min_group_size=5, cutoff=0.005)
        self.assertTrue(bool(sex_cut.quarantined_mask[18]))
        self.assertTrue(bool(sex_cut.quarantined_mask[19]))

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                entries=entries,
                prefix=Path("ref_all_m"),
                pca_components=2,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        self.assertIsNotNone(meta.sex_inference)
        if meta.sex_inference is None:
            self.fail("expected non-None meta.sex_inference")
        self.assertEqual(meta.sex_inference.get("quarantined_indices"), [18, 19])

    def test_single_sex_all_female_labeled_quarantines_xo_and_ignores_chry_xtr(
        self,
    ) -> None:
        karyotypes = [("XX", 2.0, 0.0, "F")] * 19 + [("XO", 1.0, 0.0, "F")]
        cohort, usable, entries = _simulate_human_sex_cohort(
            karyotypes, seed=202, xtr_mismap_frac=0.18
        )
        sex_inf = infer_germline_sex_2d(
            cohort,
            usable,
            min_group_size=5,
            explicit_labels=tuple("female" for _ in entries),
        )
        self.assertTrue(sex_inf.passed)
        self.assertFalse(bool(np.any(sex_inf.quarantined_mask[:19])))
        self.assertTrue(bool(sex_inf.quarantined_mask[19]))
        max_f_frac = float(np.max(np.asarray(sex_inf.fractions)[:19]))
        self.assertLess(max_f_frac, 0.005)
        self.assertIsNotNone(sex_inf.cutoff)
        if sex_inf.cutoff is None:
            self.fail("expected non-None sex_inf.cutoff")
        self.assertGreater(sex_inf.cutoff, max_f_frac * 2.5)

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                entries=entries,
                prefix=Path("ref_all_f"),
                pca_components=2,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        self.assertIsNotNone(meta.sex_inference)
        if meta.sex_inference is None:
            self.fail("expected non-None meta.sex_inference")
        self.assertEqual(meta.sex_inference.get("quarantined_indices"), [19])

    def test_mixed_labeled_cohort_quarantines_sex_label_swaps(self) -> None:
        karyotypes = (
            [("XX", 2.0, 0.0, "F")] * 10
            + [("XY", 1.0, 1.0, "M")] * 10
            + [("XY_as_F", 1.0, 1.0, "F"), ("XX_as_M", 2.0, 0.0, "M")]
        )
        cohort, usable, entries = _simulate_human_sex_cohort(karyotypes, seed=203)
        explicit = tuple("male" if e.sex == "M" else "female" for e in entries)
        sex_inf = infer_germline_sex_2d(
            cohort, usable, min_group_size=5, explicit_labels=explicit
        )
        self.assertTrue(sex_inf.passed)
        self.assertFalse(bool(np.any(sex_inf.quarantined_mask[:20])))
        self.assertTrue(bool(sex_inf.quarantined_mask[20]))
        self.assertTrue(bool(sex_inf.quarantined_mask[21]))
        self.assertIsNotNone(sex_inf.cutoff)
        if sex_inf.cutoff is None:
            self.fail("expected non-None sex_inf.cutoff")
        fracs = np.asarray(sex_inf.fractions)
        self.assertGreater(sex_inf.cutoff, float(np.max(fracs[:10])))
        self.assertLess(sex_inf.cutoff, float(np.min(fracs[10:20])))

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                entries=entries,
                prefix=Path("ref_swapped"),
                pca_components=2,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        self.assertIsNotNone(meta.sex_inference)
        if meta.sex_inference is None:
            self.fail("expected non-None meta.sex_inference")
        self.assertEqual(meta.sex_inference.get("quarantined_indices"), [20, 21])

    def test_infer_germline_sex_2d_streams_memmap_counts_chunked(self) -> None:
        karyotypes = (
            [("XX", 2.0, 0.0, "F")] * 6
            + [("XY", 1.0, 1.0, "M")] * 6
            + [("XXY", 2.0, 1.0, "M")]
        )
        cohort_ram, usable, _ = _simulate_human_sex_cohort(karyotypes, seed=204)
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = Path(tmpdir) / "cohort_counts.mmap"
            mmap_w = np.memmap(
                mmap_path,
                dtype=np.float32,
                mode="w+",
                shape=cohort_ram.counts.shape,
                order="F",
            )
            mmap_w[:] = cohort_ram.counts
            mmap_w.flush()
            del mmap_w

            mmap_ro = np.memmap(
                mmap_path,
                dtype=np.float32,
                mode="r",
                shape=cohort_ram.counts.shape,
                order="F",
            )
            cohort_mmap = Cohort(
                counts=mmap_ro,
                contigs=cohort_ram.contigs,
                starts=cohort_ram.starts,
                ends=cohort_ram.ends,
                contig_lengths=cohort_ram.contig_lengths,
                bin_size=cohort_ram.bin_size,
                identities=cohort_ram.identities,
                paths=cohort_ram.paths,
            )
            inf_ram = infer_germline_sex_2d(cohort_ram, usable, min_group_size=5)
            inf_mmap = infer_germline_sex_2d(cohort_mmap, usable, min_group_size=5)
            self.assertEqual(inf_mmap.labels, inf_ram.labels)
            self.assertAlmostEqual(
                float(inf_mmap.cutoff or 0.0), float(inf_ram.cutoff or 0.0), places=12
            )
            np.testing.assert_array_equal(
                inf_mmap.quarantined_mask, inf_ram.quarantined_mask
            )
            np.testing.assert_allclose(
                _sum_counts_by_sample(mmap_ro, usable, chunk_rows=32),
                np.sum(cohort_ram.counts[usable], axis=0, dtype=np.float64),
                rtol=1e-12,
            )
            del cohort_mmap, mmap_ro

    def test_single_sex_labeled_cohort_with_swapped_label_preserves_clean_cutoff(
        self,
    ) -> None:
        m_karyo = [("XY", 1.0, 1.0, "M")] * 19 + [("XY_as_F", 1.0, 1.0, "F")]
        m_cohort, m_usable, m_entries = _simulate_human_sex_cohort(m_karyo, seed=205)
        m_explicit = tuple("male" if e.sex == "M" else "female" for e in m_entries)
        m_inf = infer_germline_sex_2d(
            m_cohort, m_usable, min_group_size=5, explicit_labels=m_explicit
        )
        self.assertTrue(m_inf.passed)
        self.assertFalse(bool(np.any(m_inf.quarantined_mask[:19])))
        self.assertTrue(bool(m_inf.quarantined_mask[19]))
        self.assertIsNotNone(m_inf.cutoff)
        if m_inf.cutoff is None:
            self.fail("expected non-None m_inf.cutoff")
        m_fracs = np.asarray(m_inf.fractions)[:19]
        self.assertLess(m_inf.cutoff, float(np.min(m_fracs)) * 0.7)
        self.assertTrue(bool(np.all(m_fracs > m_inf.cutoff)))

        f_karyo = [("XX", 2.0, 0.0, "F")] * 19 + [("XX_as_M", 2.0, 0.0, "M")]
        f_cohort, f_usable, f_entries = _simulate_human_sex_cohort(f_karyo, seed=206)
        f_explicit = tuple("male" if e.sex == "M" else "female" for e in f_entries)
        f_inf = infer_germline_sex_2d(
            f_cohort, f_usable, min_group_size=5, explicit_labels=f_explicit
        )
        self.assertTrue(f_inf.passed)
        self.assertFalse(bool(np.any(f_inf.quarantined_mask[:19])))
        self.assertTrue(bool(f_inf.quarantined_mask[19]))
        self.assertIsNotNone(f_inf.cutoff)
        if f_inf.cutoff is None:
            self.fail("expected non-None f_inf.cutoff")
        f_fracs = np.asarray(f_inf.fractions)[:19]
        self.assertGreater(f_inf.cutoff, float(np.max(f_fracs)) * 2.5)
        self.assertTrue(bool(np.all(f_fracs <= f_inf.cutoff)))

    def test_female_sex_branch_excludes_chry_in_germline_and_prenatal_prediction(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        bin_size = 5000
        lengths = {
            "chr1": 200_000,
            "chr2": 200_000,
            "chrX": 200_000,
            "chrY": 100_000,
        }
        n_auto, n_x, n_y = 40, 40, 20

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            g_paths: list[Path] = []
            g_entries: list[SamplesheetEntry] = []
            for idx in range(12):
                is_male = idx >= 6
                c1 = rng.poisson(120, size=n_auto).astype(np.uint32)
                c2 = rng.poisson(120, size=n_auto).astype(np.uint32)
                cx = rng.poisson(60 if is_male else 120, size=n_x).astype(np.uint32)
                cy = rng.poisson(55 if is_male else 1, size=n_y).astype(np.uint32)
                counts = np.concatenate([c1, c2, cx, cy])
                s_path = tmp / f"germ_{idx}.ccnv"
                write_sample_parquet(
                    _make_sex_sample_buffers(
                        lengths, bin_size, counts, sample_id=f"G{idx}"
                    ).to_table(),
                    s_path,
                )
                g_paths.append(s_path)
                g_entries.append(
                    SamplesheetEntry(filepath=s_path, sex="M" if is_male else "F")
                )

            g_cohort = load_and_assemble_cohort(tuple(g_entries), bin_size)
            g_ref = build_reference_table(
                g_cohort,
                ReferenceConfig(
                    entries=tuple(g_entries),
                    prefix=tmp / "germ_ref",
                    min_cohort_size=6,
                    min_sex_group_size=4,
                ),
            )
            g_ref_path = tmp / "germ_ref.ccnv"
            write_reference_parquet(g_ref, g_ref_path)

            fem_rows = g_ref.filter(
                pa.compute.equal(g_ref["branch"], BRANCH_FEMALE_SEX)
            )
            male_rows = g_ref.filter(pa.compute.equal(g_ref["branch"], BRANCH_MALE_SEX))
            self.assertEqual(
                set(decode_dictionary_contigs(fem_rows.column("contig"))), {"chrX"}
            )
            self.assertEqual(
                set(decode_dictionary_contigs(male_rows.column("contig"))),
                {"chrX", "chrY"},
            )

            out_xx = run_predict(
                PredictConfig(
                    sample=g_paths[0],
                    reference=g_ref_path,
                    prefix=tmp / "pred_xx",
                    sex="female",
                    plot=False,
                )
            )
            self.assertEqual([s for s in out_xx.segments if s.call is not None], [])
            fem_bin_res = next(b for b in out_xx.bins if b.branch == BRANCH_FEMALE_SEX)
            self.assertEqual(set(fem_bin_res.contigs), {"chrX"})
            self.assertGreater(
                int(np.count_nonzero(np.isfinite(fem_bin_res.ratios))), 30
            )

            p_paths: list[Path] = []
            p_entries: list[SamplesheetEntry] = []
            for idx in range(12):
                is_male_fetus = idx >= 6
                ff = 0.08 + 0.015 * (idx - 6) if is_male_fetus else 0.0
                c1 = rng.poisson(200, size=n_auto).astype(np.uint32)
                c2 = rng.poisson(200, size=n_auto).astype(np.uint32)
                cx = rng.poisson(200.0 * (1.0 - 0.5 * ff), size=n_x).astype(np.uint32)
                cy = rng.poisson(
                    max(1.0, 100.0 * ff) if is_male_fetus else 0.3, size=n_y
                ).astype(np.uint32)
                counts = np.concatenate([c1, c2, cx, cy])
                s_path = tmp / f"pren_{idx}.ccnv"
                write_sample_parquet(
                    _make_sex_sample_buffers(
                        lengths,
                        bin_size,
                        counts,
                        sample_id=f"P{idx}",
                        with_fragmentomics=True,
                        fetal_like=is_male_fetus,
                    ).to_table(),
                    s_path,
                )
                p_paths.append(s_path)
                p_entries.append(SamplesheetEntry(filepath=s_path))

            p_cohort = load_and_assemble_cohort(tuple(p_entries), bin_size)
            p_ref = build_reference_table(
                p_cohort,
                ReferenceConfig(
                    entries=tuple(p_entries),
                    prefix=tmp / "pren_ref",
                    mode="prenatal",
                    prenatal_bin_size=50_000,
                    min_cohort_size=6,
                    min_sex_group_size=4,
                ),
            )
            p_ref_path = tmp / "pren_ref.ccnv"
            write_reference_parquet(p_ref, p_ref_path)

            out_xy = run_predict(
                PredictConfig(
                    sample=p_paths[9],
                    reference=p_ref_path,
                    prefix=tmp / "pred_pren_xy",
                    mode="prenatal",
                    plot=False,
                )
            )
            self.assertEqual([s for s in out_xy.segments if s.call is not None], [])
            self.assertNotIn(
                "chrY", {s.contig for s in out_xy.segments if s.discordant}
            )


if __name__ == "__main__":
    unittest.main()
