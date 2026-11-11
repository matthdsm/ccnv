"""Tests for geometric exposure, 1 Mb super-block size factors, per-contig masking, and 2D sex gating."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_sample_table

from ccnv.features.cohort import (
    Cohort,
    InvalidInputError,
    compute_bin_exposure,
    estimate_superblock_size_factors,
    infer_germline_sex_2d,
    mask_low_coverage_per_contig,
    normalize_depth,
)
from ccnv.io.parquet import write_reference_parquet, write_sample_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    ReferenceConfig,
    ReferenceMetadata,
    SamplesheetEntry,
)
from ccnv.predict import load_inputs
from ccnv.reference import build_reference_table


class TestSuperblockDepthAndSex(unittest.TestCase):
    def test_compute_bin_exposure_handles_partial_terminal_bins_and_validates(
        self,
    ) -> None:
        starts = np.array([0, 1000, 2000], dtype=np.uint32)
        ends = np.array([1000, 2000, 2350], dtype=np.uint32)
        exp = compute_bin_exposure(starts, ends, 1000)
        self.assertEqual(exp.dtype, np.float32)
        np.testing.assert_allclose(exp, [1.0, 1.0, 0.35], rtol=1e-6)

        with self.assertRaises(InvalidInputError):
            compute_bin_exposure(starts, ends, 0)
        with self.assertRaises(InvalidInputError):
            compute_bin_exposure(
                np.array([500], dtype=np.uint32),
                np.array([400], dtype=np.uint32),
                1000,
            )

    def test_superblock_size_factors_unquantized_at_01x_1kb_s200(self) -> None:
        rng = np.random.default_rng(42)
        bin_size = 1_000
        n_contigs = 12
        bins_per_contig = 12_000  # 12 Mb per contig -> 144 Mb total
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
            usable[base + 5500 : base + 6500] = False  # 1 Mb centromere gap

        # Simulate S=200 samples at 1 kb bins with true depths spanning 0.10x..0.35x (lambda=0.33..1.16)
        true_depths = np.linspace(0.10, 0.35, n_samples, dtype=np.float64)
        lambdas = (true_depths / 0.30) * 1.0  # 0.333 .. 1.167 reads per full 1 kb bin
        rate_matrix = (
            exposure[:, np.newaxis].astype(np.float64) * lambdas[np.newaxis, :]
        )
        # Inject 5 Mb 4x amplification on contig 0 in 10 samples to test CNV robustness
        rate_matrix[:5_000, :10] *= 4.0
        counts = rng.poisson(rate_matrix).astype(np.float32)
        counts[~usable, :] = 0.0

        sf = estimate_superblock_size_factors(counts, contig_ids, starts, ends, usable)
        self.assertEqual(sf.dtype, np.float64)
        self.assertAlmostEqual(float(np.median(sf)), 1.0, places=12)
        self.assertEqual(np.unique(np.round(sf, 6)).size, n_samples)

        rel_err = np.abs(sf / np.median(sf) - true_depths / np.median(true_depths)) / (
            true_depths / np.median(true_depths)
        )
        self.assertLess(float(np.max(rel_err)), 0.04)

        normed = normalize_depth(
            counts[:, :8],
            mask=usable,
            contigs=contig_ids,
            starts=starts,
            ends=ends,
        )
        # Must stay on O(lambda) read-count scale (~0.35 reads/bin), never 1e-5 library fraction
        diploid_usable = usable & (contig_ids > 0)
        self.assertGreater(float(np.mean(normed[diploid_usable, 0])), 0.25)
        self.assertLess(float(np.mean(normed[diploid_usable, 0])), 0.50)

        zero_col = counts[:, :4].copy()
        zero_col[:, 2] = 0.0
        with self.assertRaises(InvalidInputError):
            estimate_superblock_size_factors(zero_col, contig_ids, starts, ends, usable)

    def test_per_contig_coverage_mask_preserves_clean_prenatal_chry_at_8pct_ff(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        bin_size = 5_000
        n_auto = 400
        n_chry = 200
        n_bins = n_auto + n_chry
        n_samples = 24  # 12 female pregnancies (0% chrY) + 12 male pregnancies (8% FF)

        contig_ids = np.concatenate(
            [
                np.zeros(n_auto, dtype=np.int16),
                np.ones(n_chry, dtype=np.int16),
            ]
        )
        starts = np.concatenate(
            [
                np.arange(n_auto, dtype=np.uint32) * np.uint32(bin_size),
                np.arange(n_chry, dtype=np.uint32) * np.uint32(bin_size),
            ]
        )
        ends = starts + np.uint32(bin_size)
        # Make final chrY bin a partial 1000 bp bin (exposure = 0.2)
        ends[-1] = starts[-1] + np.uint32(1_000)
        exposure = compute_bin_exposure(starts, ends, bin_size)

        counts_prenatal_8pct_ff = np.zeros((n_bins, n_samples), dtype=np.float32)
        auto_lambda = 200.0
        counts_prenatal_8pct_ff[:n_auto, :] = rng.poisson(
            auto_lambda, size=(n_auto, n_samples)
        ).astype(np.float32)

        # At 8% fetal fraction, haploid MSY rate in male pregnancies is 0.04 * auto_lambda = 8.0 reads/bin
        # Cohort-wide chrY sum across 12 XX + 12 XY is ~96 vs autosomal ~4800 (2% < 5% floor!)
        chry_slice = slice(n_auto, n_bins)
        chry_clean_local = np.zeros(n_chry, dtype=np.bool_)
        chry_clean_local[10:120] = True
        chry_clean_local[-1] = True  # partial terminal bin with healthy rate
        chry_clean_mask = np.zeros(n_bins, dtype=np.bool_)
        chry_clean_mask[chry_slice] = chry_clean_local

        for idx in np.flatnonzero(chry_clean_mask):
            counts_prenatal_8pct_ff[idx, 12:] = rng.poisson(
                0.04 * auto_lambda * float(exposure[idx]), size=12
            ).astype(np.float32)

        # Heterochromatic dead region & dead telomere bin on chrY
        chry_dead_telomere_idx = n_auto + 0
        counts_prenatal_8pct_ff[chry_dead_telomere_idx, :] = 0.0
        # Ultra-low mismapping bin (< 5% of chrY positive median: 1 read across 24 samples vs ~96)
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
        rng = np.random.default_rng(123)
        bin_size = (
            500_000  # 500 kb bins on human-scale contigs to exercise PAR exclusion
        )
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

        # 10 XX, 10 XY, 1 XXY, 1 mLOY (35% Y), 1 XO, 1 XYY -> 24 samples
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
        xxy_idx = 20
        mloy_idx = 21
        xo_idx = 22
        xyy_idx = 23
        n_samples = len(karyotypes)

        is_auto = np.isin(contigs_arr, ["chr1", "chr2"])
        is_x = contigs_arr == "chrX"
        is_y = contigs_arr == "chrY"
        is_x_par = is_x & ((starts_arr < 2_500_000) | (ends_arr > 59_500_000))
        is_y_par = is_y & ((starts_arr < 2_500_000) | (ends_arr > 57_500_000))
        is_y_het = is_y & (starts_arr >= 28_000_000) & (ends_arr <= 57_000_000)

        counts = np.zeros((n_bins, n_samples), dtype=np.float32)
        tables = []
        for s_idx, (_, ploidy_x, ploidy_y, _) in enumerate(karyotypes):
            diploid_mu = 400.0 * rng.uniform(0.8, 1.25)
            mu = np.zeros(n_bins, dtype=np.float64)
            mu[is_auto] = diploid_mu
            # PAR1/PAR2 behave as diploid (2 copies) regardless of XX/XY!
            mu[is_x_par] = diploid_mu
            mu[is_y_par] = diploid_mu * 0.5
            mu[is_x & ~is_x_par] = diploid_mu * (ploidy_x / 2.0)
            mu[is_y & ~is_y_par & ~is_y_het] = (
                diploid_mu * (ploidy_y / 2.0) + diploid_mu * 0.002
            )
            sample_counts = rng.poisson(mu).astype(np.uint32)
            counts[:, s_idx] = sample_counts.astype(np.float32)
            tables.append(
                make_sample_table(
                    sample_counts.tolist(),
                    contigs=contigs_list,
                    starts=starts_list,
                    ends=ends_list,
                    contig_lengths=contig_lengths,
                    bin_size=bin_size,
                    identity={"id": f"s_{s_idx}"},
                )
            )

        cohort_with_xxy_and_mloy = Cohort(
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

        sex_inf = infer_germline_sex_2d(
            cohort_with_xxy_and_mloy, usable, min_group_size=5
        )
        self.assertTrue(sex_inf.passed)
        self.assertTrue(bool(sex_inf.quarantined_mask[xxy_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[mloy_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[xo_idx]))
        self.assertTrue(bool(sex_inf.quarantined_mask[xyy_idx]))
        self.assertFalse(bool(np.any(sex_inf.quarantined_mask[:20])))
        assert sex_inf.cutoff is not None
        self.assertGreater(sex_inf.cutoff, 0.0)
        self.assertEqual(sex_inf.chry_cutoff, sex_inf.cutoff)
        self.assertEqual(sex_inf.sample_sexes, sex_inf.labels)

        # Verify reference building excludes quarantined samples from sex branches AND persists cutoff when labeled
        labeled_entries = tuple(
            SamplesheetEntry(filepath=Path(f"s_{i}.ccnv"), sex=sex_lbl)
            for i, (_, _, _, sex_lbl) in enumerate(karyotypes)
        )
        ref_unlabeled = build_reference_table(
            cohort_with_xxy_and_mloy,
            ReferenceConfig(
                samples=tuple(e.filepath for e in labeled_entries),
                prefix=Path("ref_unlabeled"),
                pca_components=2,
                nearest_bins=4,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        ref_labeled = build_reference_table(
            cohort_with_xxy_and_mloy,
            ReferenceConfig(
                samples=tuple(e.filepath for e in labeled_entries),
                entries=labeled_entries,
                prefix=Path("ref_labeled"),
                pca_components=2,
                nearest_bins=4,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )

        meta_labeled = ReferenceMetadata.from_table(ref_labeled)
        self.assertEqual(meta_labeled.sex_source, "samplesheet")
        self.assertIsNotNone(meta_labeled.sex_cutoff)
        assert meta_labeled.sex_cutoff is not None
        self.assertGreater(meta_labeled.sex_cutoff, 0.0)
        self.assertIsNotNone(meta_labeled.sex_inference)
        assert meta_labeled.sex_inference is not None
        self.assertIsNotNone(meta_labeled.sex_inference.get("cutoff"))

        # Check male_sex chrX/chrY ratio on non-PAR bins: must reflect pure 1:1 XY, uncorrupted by XXY/mLOY/XYY
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

        # Verify predict load_inputs auto-sex (--sex autosomal) works against the labeled reference!
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref_path = root / "ref_labeled.ccnv"
            xx_path = root / "xx_sample.ccnv"
            xy_path = root / "xy_sample.ccnv"
            write_reference_parquet(ref_labeled, ref_path)
            write_sample_parquet(tables[0], xx_path)
            write_sample_parquet(tables[10], xy_path)

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
        self.assertTrue(bool(sex_inf.quarantined_mask[18]))  # XXY
        self.assertTrue(bool(sex_inf.quarantined_mask[19]))  # mLOY
        assert sex_inf.cutoff is not None
        self.assertLess(
            sex_inf.cutoff, float(np.min(np.asarray(sex_inf.fractions)[:18]))
        )

        sex_cut = infer_germline_sex_2d(cohort, usable, min_group_size=5, cutoff=0.005)
        self.assertTrue(bool(sex_cut.quarantined_mask[18]))
        self.assertTrue(bool(sex_cut.quarantined_mask[19]))

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                samples=tuple(e.filepath for e in entries),
                entries=entries,
                prefix=Path("ref_all_m"),
                pca_components=2,
                nearest_bins=4,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        assert meta.sex_inference is not None
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
        self.assertTrue(bool(sex_inf.quarantined_mask[19]))  # XO
        self.assertLess(float(np.max(np.asarray(sex_inf.d_y)[:19])), 0.05)
        assert sex_inf.cutoff is not None
        max_f_frac = float(np.max(np.asarray(sex_inf.fractions)[:19]))
        self.assertGreater(sex_inf.cutoff, max_f_frac * 2.5)

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                samples=tuple(e.filepath for e in entries),
                entries=entries,
                prefix=Path("ref_all_f"),
                pca_components=2,
                nearest_bins=4,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        assert meta.sex_inference is not None
        self.assertEqual(meta.sex_inference.get("quarantined_indices"), [19])

    def test_mixed_labeled_cohort_quarantines_sex_label_swaps(self) -> None:
        # 10 XX labeled F, 10 XY labeled M, 1 XY mislabeled F (idx 20), 1 XX mislabeled M (idx 21)
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
        self.assertTrue(bool(sex_inf.quarantined_mask[20]))  # XY swapped to F
        self.assertTrue(bool(sex_inf.quarantined_mask[21]))  # XX swapped to M
        assert sex_inf.cutoff is not None
        fracs = np.asarray(sex_inf.fractions)
        self.assertGreater(sex_inf.cutoff, float(np.max(fracs[:10])))
        self.assertLess(sex_inf.cutoff, float(np.min(fracs[10:20])))

        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                samples=tuple(e.filepath for e in entries),
                entries=entries,
                prefix=Path("ref_swapped"),
                pca_components=2,
                nearest_bins=4,
                min_cohort_size=5,
                min_sex_group_size=5,
            ),
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        assert meta.sex_inference is not None
        self.assertEqual(meta.sex_inference.get("quarantined_indices"), [20, 21])

    def test_infer_germline_sex_2d_streams_memmap_counts_chunked(self) -> None:
        from ccnv.features.sex import _sum_counts_by_sample

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
        # 19 XY labeled M + 1 XY mislabeled F -> quarantined swap must not pollute cf/cutoff
        m_karyo = [("XY", 1.0, 1.0, "M")] * 19 + [("XY_as_F", 1.0, 1.0, "F")]
        m_cohort, m_usable, m_entries = _simulate_human_sex_cohort(m_karyo, seed=205)
        m_explicit = tuple("male" if e.sex == "M" else "female" for e in m_entries)
        m_inf = infer_germline_sex_2d(
            m_cohort, m_usable, min_group_size=5, explicit_labels=m_explicit
        )
        self.assertTrue(m_inf.passed)
        self.assertFalse(bool(np.any(m_inf.quarantined_mask[:19])))
        self.assertTrue(bool(m_inf.quarantined_mask[19]))
        assert m_inf.cutoff is not None
        m_fracs = np.asarray(m_inf.fractions)[:19]
        self.assertLess(m_inf.cutoff, float(np.min(m_fracs)) * 0.7)
        self.assertTrue(bool(np.all(m_fracs > m_inf.cutoff)))

        # 19 XX labeled F + 1 XX mislabeled M -> quarantined swap must not pollute cm/cutoff
        f_karyo = [("XX", 2.0, 0.0, "F")] * 19 + [("XX_as_M", 2.0, 0.0, "M")]
        f_cohort, f_usable, f_entries = _simulate_human_sex_cohort(f_karyo, seed=206)
        f_explicit = tuple("male" if e.sex == "M" else "female" for e in f_entries)
        f_inf = infer_germline_sex_2d(
            f_cohort, f_usable, min_group_size=5, explicit_labels=f_explicit
        )
        self.assertTrue(f_inf.passed)
        self.assertFalse(bool(np.any(f_inf.quarantined_mask[:19])))
        self.assertTrue(bool(f_inf.quarantined_mask[19]))
        assert f_inf.cutoff is not None
        f_fracs = np.asarray(f_inf.fractions)[:19]
        self.assertGreater(f_inf.cutoff, float(np.max(f_fracs)) * 2.5)
        self.assertTrue(bool(np.all(f_fracs <= f_inf.cutoff)))


def _simulate_human_sex_cohort(
    karyotypes: list[tuple[str, float, float, str]],
    *,
    seed: int,
    xtr_mismap_frac: float = 0.002,
) -> tuple[Cohort, np.ndarray, tuple[SamplesheetEntry, ...]]:
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
    is_y_xtr = is_y & (starts_arr >= 4_000_000) & (ends_arr <= 6_000_000)  # 2 Mb XTR

    counts = np.zeros((n_bins, n_samples), dtype=np.float32)
    for s_idx, (_, ploidy_x, ploidy_y, _) in enumerate(karyotypes):
        diploid_mu = 400.0 * rng.uniform(0.85, 1.20)
        mu = np.zeros(n_bins, dtype=np.float64)
        mu[is_auto] = diploid_mu
        mu[is_x_par] = diploid_mu
        mu[is_y_par] = 0.0
        mu[is_x & ~is_x_par] = diploid_mu * (ploidy_x / 2.0)
        mu[is_y & ~is_y_par & ~is_y_het] = diploid_mu * (ploidy_y / 2.0)
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


if __name__ == "__main__":
    unittest.main()
