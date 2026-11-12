"""Regression tests for Round 5 pipeline, biology, math, I/O, and contract fixes."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyBigWig
import pysam
from numpy.typing import NDArray

from ccnv.features.binning import bin_intervals, rebin_counts
from ccnv.features.cohort import (
    load_and_assemble_cohort,
    rebin_cohort,
    validate_cohort,
)
from ccnv.features.prenatal_origin import (
    BandPrefixSums,
    annotate_prenatal_origin_and_lod,
)
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.alignments import scan_alignment_bins
from ccnv.io.parquet import (
    read_reference_parquet,
    read_sample_buffers,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.io.tracks import load_mappability_track
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    COORDINATE_SYSTEM,
    NUM_FRAG_BANDS,
    NUM_PROTOTYPES_PER_BIN,
    SCHEMA_VERSION,
    AlignmentError,
    BinResult,
    BranchFitArrays,
    CcnvError,
    CovariateError,
    FetalFractionResult,
    FragmentomicsMetadata,
    InvalidInputError,
    PredictConfig,
    PrenatalModelMetadata,
    ReadFilters,
    ReferenceConfig,
    ReferenceError,
    ReferenceMetadata,
    SampleBuffers,
    SampleMetadata,
    SamplesheetEntry,
    SCAResult,
    SchemaError,
    SegmentResult,
    assemble_reference_table,
    decode_dictionary_contigs,
    decode_metadata,
    encode_metadata,
)
from ccnv.predict import run as run_predict
from ccnv.qc.reference import run_reference_qc_on_table
from ccnv.reference import build_reference_table
from ccnv.report.html import render_genome_plot
from ccnv.stats.pca import project_huber_pca_wave
from ccnv.stats.regression import fit_logit_linear_ensemble
from ccnv.stats.segmentation import segment_contig


def _make_sample_buffers(
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
        frag_len_sum = (counts.astype(np.uint64)) * np.uint64(
            162 if fetal_like else 168
        )
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
        motifs = [10] * 256
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
            end_motif_counts=motifs,
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


class TestRound5DefectRegressions(unittest.TestCase):
    def test_picard_multi_dot_bam_index_resolution_ignores_sibling_stem_bai(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bam_path = tmp / "sample.sorted.bam"
            header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 20_000}]}
            with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out_bam:
                for idx in range(25):
                    rec = pysam.AlignedSegment(out_bam.header)
                    rec.query_name = f"r{idx}"
                    rec.query_sequence = "ACGT" * 25
                    rec.flag = 0
                    rec.reference_id = 0
                    rec.reference_start = idx * 500 + 50
                    rec.mapping_quality = 60
                    rec.cigar = ((0, 100),)
                    rec.query_qualities = pysam.qualitystring_to_array("I" * 100)
                    out_bam.write(rec)
            pysam.index(str(bam_path))

            samtools_bai = tmp / "sample.sorted.bam.bai"
            picard_bai = tmp / "sample.sorted.bai"
            decoy_bai = tmp / "sample.bai"
            shutil.move(str(samtools_bai), str(picard_bai))
            decoy_bai.write_bytes(b"NOT_A_VALID_BAI_INDEX")

            filters = ReadFilters(min_mapping_quality=1, require_proper_pair=False)
            res_1 = scan_alignment_bins(
                bam_path,
                (("chr1", 20_000),),
                bin_size=5000,
                filters=filters,
                threads=1,
            )
            res_2 = scan_alignment_bins(
                bam_path,
                (("chr1", 20_000),),
                bin_size=5000,
                filters=filters,
                threads=2,
            )
            np.testing.assert_array_equal(res_1["count"], res_2["count"])

            picard_bai.unlink()
            with self.assertRaises(AlignmentError):
                scan_alignment_bins(
                    bam_path,
                    (("chr1", 20_000),),
                    bin_size=5000,
                    filters=filters,
                    threads=2,
                )

    def test_load_mappability_track_rejects_percent_scaled_bigwig(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_bw_path = tmp / "percent_map.bw"
            bw = pyBigWig.open(str(bad_bw_path), "w")
            bw.addHeader([("chr1", 10_000)])
            bw.addEntries(["chr1"], [0], ends=[10_000], values=[95.0])
            bw.close()

            contigs = ("chr1", "chr1")
            starts = np.array([0, 5000], dtype=np.int64)
            ends = np.array([5000, 10_000], dtype=np.int64)
            with self.assertRaisesRegex(CovariateError, r"outside \[0\.0, 1\.0\]"):
                load_mappability_track(bad_bw_path, contigs, starts, ends)

            good_bw_path = tmp / "unit_map.bw"
            bw_ok = pyBigWig.open(str(good_bw_path), "w")
            bw_ok.addHeader([("chr1", 10_000)])
            bw_ok.addEntries(["chr1"], [0], ends=[5000], values=[0.85])
            bw_ok.close()
            loaded = load_mappability_track(good_bw_path, contigs, starts, ends)
            np.testing.assert_allclose(loaded, np.array([0.85, 0.0]), atol=1e-6)

    def test_parquet_writers_are_atomic_and_preserve_existing_file_on_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lengths = {"chr1": 10_000}
            bufs_v1 = _make_sample_buffers(
                lengths, 5000, np.array([100, 120], dtype=np.uint32), sample_id="V1"
            )
            bufs_v2 = _make_sample_buffers(
                lengths, 5000, np.array([999, 888], dtype=np.uint32), sample_id="V2"
            )
            s_path = tmp / "sample.ccnv"
            write_sample_parquet(bufs_v1.to_table(), s_path)

            with (
                patch("pyarrow.parquet.write_table", side_effect=OSError("disk full")),
                self.assertRaises(OSError),
            ):
                write_sample_parquet(bufs_v2.to_table(), s_path)

            reloaded_s = read_sample_buffers(s_path)
            np.testing.assert_array_equal(
                reloaded_s.count, np.array([100, 120], dtype=np.uint32)
            )
            self.assertEqual(list(tmp.glob("*.tmp")), [])

            fit = BranchFitArrays(
                branch=BRANCH_AUTOSOMAL,
                contigs=("chr1", "chr1"),
                starts=np.array([0, 5000], dtype=np.int64),
                ends=np.array([5000, 10_000], dtype=np.int64),
                masked=np.array([False, False], dtype=np.bool_),
                gc=None,
                mappability=None,
                gc_quantiles=None,
                ref_frag_len=None,
                ref_ep_rate=None,
                pca_mean=np.array([100.0, 100.0], dtype=np.float32),
                pca_loadings=np.zeros((2, 0), dtype=np.float32),
                proto_ids=np.zeros((2, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16),
                proto_weights=np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (2, 1)
                ),
                mean_distance=np.zeros(2, dtype=np.float32),
                null_mean=np.zeros(2, dtype=np.float32),
                bio_dispersion=np.full(2, 0.01, dtype=np.float32),
            )
            ref_meta = ReferenceMetadata(
                bin_size=5000,
                contig_lengths=lengths,
                branches=(BRANCH_AUTOSOMAL,),
                autosomal_contigs=("chr1",),
                sex_contigs=(),
                pca_components=0,
            )
            ref_tbl = assemble_reference_table(
                [fit], contig_order=("chr1",), metadata=ref_meta.build_arrow_metadata()
            )
            r_path = tmp / "ref.ccnv"
            write_reference_parquet(ref_tbl, r_path)

            with (
                patch(
                    "pyarrow.parquet.ParquetWriter.write_table",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaises(OSError),
            ):
                write_reference_parquet(ref_tbl, r_path)

            reloaded_r, _ = read_reference_parquet(r_path)
            self.assertEqual(reloaded_r.num_rows, 2)
            self.assertEqual(list(tmp.glob("*.tmp")), [])

    def test_rfc8259_metadata_and_html_reject_or_sanitize_non_finite_floats(
        self,
    ) -> None:
        with self.assertRaises(SchemaError):
            encode_metadata({"bad": {"x": float("nan")}})
        with self.assertRaises(SchemaError):
            decode_metadata({b"ccnv.bad": b'{"x":NaN}'})
        with self.assertRaises(SchemaError):
            PrenatalModelMetadata.decode(b'{"prenatal_bin_size":100000,"x":Infinity}')

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_plot.html"
            bins = (
                BinResult(
                    branch=BRANCH_AUTOSOMAL,
                    contigs=("chr1", "chr1"),
                    starts=np.array([0, 5000], dtype=np.int64),
                    ends=np.array([5000, 10_000], dtype=np.int64),
                    counts=np.array([50.0, 52.0]),
                    masked=np.array([False, False]),
                    ratios=np.array([0.0, 0.02]),
                    zscores=np.array([0.0, 0.2]),
                    weights=np.array([25.0, 25.0]),
                    reference_sizes=np.array([10, 10], dtype=np.int64),
                    expected_counts=np.array([50.0, 50.0]),
                ),
            )
            ff_res = FetalFractionResult(
                fetal_sex="F",
                active_ff=float("nan"),
                active_ff_source="FFA",
                ffy=None,
                ffx=float("inf"),
                ffa=0.08,
                chrx_ratio=1.0,
                chry_score=0.0,
                yfrac_threshold=0.001,
                sca=SCAResult(
                    call="XX", is_aneuploid=False, confidence=0.99, chrx_zscore=0.0
                ),
                low_ff_warning=True,
                warnings=(),
                ff_size=float("-inf"),
            )
            render_genome_plot(
                html_path,
                bins,
                (),
                {"chr1": 10_000},
                prenatal_result=ff_res,
            )
            rendered = html_path.read_text(encoding="utf-8")
            match = re.search(
                r'<script id="ccnv-data" type="application/json">(.*?)</script>',
                rendered,
                re.DOTALL,
            )
            if match is None:
                self.fail("missing ccnv-data script tag in rendered HTML")
            parsed = json.loads(
                match.group(1),
                parse_constant=lambda tok: (_ for _ in ()).throw(ValueError(tok)),
            )
            self.assertIsNone(parsed["prenatal"]["active_ff"])
            self.assertIsNone(parsed["prenatal"]["ffx"])
            self.assertIsNone(parsed["prenatal"]["ff_size"])

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
        n_auto = 40
        n_x = 40
        n_y = 20

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            g_paths: list[Path] = []
            g_entries: list[SamplesheetEntry] = []
            for idx in range(12):
                is_male = idx >= 6
                c1 = rng.poisson(120, size=n_auto).astype(np.uint32)
                c2 = rng.poisson(120, size=n_auto).astype(np.uint32)
                cx = rng.poisson(60 if is_male else 120, size=n_x).astype(np.uint32)
                cy = (
                    rng.poisson(55, size=n_y).astype(np.uint32)
                    if is_male
                    else rng.poisson(1, size=n_y).astype(np.uint32)
                )
                counts = np.concatenate([c1, c2, cx, cy])
                s_path = tmp / f"germ_{idx}.ccnv"
                write_sample_parquet(
                    _make_sample_buffers(
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
                cy = (
                    rng.poisson(max(1.0, 100.0 * ff), size=n_y).astype(np.uint32)
                    if is_male_fetus
                    else rng.poisson(0.3, size=n_y).astype(np.uint32)
                )
                counts = np.concatenate([c1, c2, cx, cy])
                s_path = tmp / f"pren_{idx}.ccnv"
                write_sample_parquet(
                    _make_sample_buffers(
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
                "chrY",
                {s.contig for s in out_xy.segments if s.discordant},
            )

    def test_explicit_samplesheet_sexes_without_chry_and_zero_row_prenatal_ffy_rejected(
        self,
    ) -> None:
        rng = np.random.default_rng(7)
        lengths = {"chr1": 100_000, "chr2": 100_000, "chrX": 100_000}
        tables: list[pa.Table] = []
        entries: list[SamplesheetEntry] = []
        for idx in range(8):
            is_male = idx >= 4
            counts = np.concatenate(
                [
                    rng.poisson(100, size=20).astype(np.uint32),
                    rng.poisson(100, size=20).astype(np.uint32),
                    rng.poisson(50 if is_male else 100, size=20).astype(np.uint32),
                ]
            )
            tables.append(
                _make_sample_buffers(
                    lengths, 5000, counts, sample_id=f"NX{idx}"
                ).to_table()
            )
            entries.append(
                SamplesheetEntry(
                    filepath=Path(f"nx_{idx}.ccnv"), sex="M" if is_male else "F"
                )
            )

        cohort = validate_cohort(tuple(tables))
        ref_tbl = build_reference_table(
            cohort,
            ReferenceConfig(
                entries=tuple(entries),
                prefix=Path("unused"),
                min_cohort_size=4,
                min_sex_group_size=3,
            ),
        )
        branches = set(ref_tbl.column("branch").to_pylist())
        self.assertEqual(
            branches, {BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX}
        )
        meta = ReferenceMetadata.from_table(ref_tbl)
        self.assertEqual(meta.sex_source, "samplesheet")

        with self.assertRaisesRegex(ReferenceError, "prenatal_ffy"):
            build_reference_table(
                cohort,
                ReferenceConfig(
                    entries=tuple(entries),
                    prefix=Path("unused_prenatal"),
                    mode="prenatal",
                    prenatal_bin_size=50_000,
                    min_cohort_size=4,
                    min_sex_group_size=3,
                ),
            )

    def test_prenatal_origin_skips_diploid_math_on_chry_segments(self) -> None:
        seg_y = SegmentResult(
            contig="chrY",
            start=0,
            end=50_000,
            ratio=3.2,
            zscore=0.5,
            weight=20.0,
            bin_count=10,
            call=None,
        )
        bands = np.tile(np.ones(NUM_FRAG_BANDS, dtype=np.uint32), (10, 1))
        prefix = BandPrefixSums(
            sample_bands=bands,
            bin_contigs=np.zeros(10, dtype=np.int16),
            contig_names=("chrY",),
            bin_starts=np.arange(10, dtype=np.int64) * 5000,
            bin_ends=(np.arange(10, dtype=np.int64) + 1) * 5000,
        )
        p_f = [1.0 / 571.0] * 571
        p_m = [1.0 / 571.0] * 571

        ann_xy, _ = annotate_prenatal_origin_and_lod(
            [seg_y],
            active_ff=0.10,
            sca_call="XY",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertIsNone(ann_xy[0].expected_ratio)
        self.assertIsNone(ann_xy[0].implied_ff)
        self.assertFalse(ann_xy[0].discordant)
        self.assertIsNone(ann_xy[0].origin_call)
        self.assertIsNotNone(ann_xy[0].frag_short_ratio)

        ann_xyy, _ = annotate_prenatal_origin_and_lod(
            [seg_y],
            active_ff=0.10,
            sca_call="XYY",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertTrue(ann_xyy[0].discordant)
        self.assertEqual(ann_xyy[0].discordance_reason, "SCA_DISCORDANT")

        seg_y_gain = SegmentResult(
            contig="chrY",
            start=0,
            end=50_000,
            ratio=2.5,
            zscore=6.0,
            weight=20.0,
            bin_count=10,
            call="gain",
        )
        ann_xx_gain, _ = annotate_prenatal_origin_and_lod(
            [seg_y_gain],
            active_ff=0.10,
            sca_call="XX",
            prefix_sums=prefix,
            p_fetal_density=p_f,
            p_maternal_density=p_m,
        )
        self.assertTrue(ann_xx_gain[0].discordant)
        self.assertEqual(ann_xx_gain[0].discordance_reason, "SCA_DISCORDANT")
        self.assertIsNone(ann_xx_gain[0].origin_call)

    def test_kernel_cpd_jump_pca_wave_and_partial_nan_logit_ensemble(self) -> None:
        starts = np.arange(12, dtype=np.int64) * 1000
        ends = starts + 1000
        ratios = np.array([0.0] * 5 + [2.5] * 7, dtype=np.float64)
        weights = np.full(12, 4.0, dtype=np.float64)
        segs = segment_contig(
            contig="chr1",
            starts=starts,
            ends=ends,
            ratios=ratios,
            weights=weights,
            contig_length=12_000,
            penalty=2.0,
            min_size=3,
            jump=3,
        )
        self.assertGreaterEqual(len(segs), 2)
        for seg in segs[:-1]:
            self.assertEqual((seg.end // 1000) % 3, 0)

        coef0, wave0 = project_huber_pca_wave(
            np.array([0.1, -0.1, 0.2]),
            np.empty((0, 0), dtype=np.float64),
        )
        self.assertEqual(coef0.shape, (0,))
        self.assertEqual(wave0.shape, (3,))

        sig = np.array([0.2, -0.2, 0.4, -0.4, 0.2, 0.0], dtype=np.float64)
        ld = np.array([[0.5, -0.5, 0.5, -0.5, 0.5, 0.0]], dtype=np.float64)
        w_scale = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float64)
        coef, wave = project_huber_pca_wave(sig, ld, whitening_scale=w_scale)
        self.assertEqual(coef.shape, (1,))
        self.assertEqual(wave.shape, (6,))
        self.assertEqual(wave[5], 0.0)
        self.assertTrue(bool(np.all(np.isfinite(wave))))

        with self.assertRaises(InvalidInputError):
            project_huber_pca_wave(
                sig,
                ld,
                eval_loadings=ld,
                whitening_scale=w_scale,
                eval_whitening_scale=None,
            )

        rng = np.random.default_rng(19)
        x_mat = rng.normal(size=(12, 3))
        true_ff = np.clip(0.12 + 0.06 * x_mat[:, 0], 0.02, 0.40)
        y_partial = true_ff.copy()
        y_partial[8:] = np.nan
        fit = fit_logit_linear_ensemble(x_mat, y_partial, standardize=True, seed=0)
        self.assertGreater(float(np.linalg.norm(fit.weights)), 0.05)

    def test_ref_qc_scales_distance_thresholds_per_branch_pca_rank(self) -> None:
        rng = np.random.default_rng(11)
        n_bins = 20
        contigs = ("chr1",) * 10 + ("chr2",) * 10
        starts = np.tile(np.arange(10, dtype=np.int64) * 5000, 2)
        ends = starts + 5000
        dists = np.linspace(1.0, 18.0, n_bins, dtype=np.float32)
        pids = np.zeros((n_bins, NUM_PROTOTYPES_PER_BIN), dtype=np.uint16)
        pids[:10, 0] = 1
        pids[10:, 0] = 0
        pweights = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (n_bins, 1)
        )
        fit = BranchFitArrays(
            branch=BRANCH_AUTOSOMAL,
            contigs=contigs,
            starts=starts,
            ends=ends,
            masked=np.zeros(n_bins, dtype=np.bool_),
            gc=None,
            mappability=None,
            gc_quantiles=None,
            ref_frag_len=None,
            ref_ep_rate=None,
            pca_mean=rng.uniform(80.0, 120.0, size=n_bins).astype(np.float32),
            pca_loadings=rng.normal(size=(n_bins, 1)).astype(np.float32),
            proto_ids=pids,
            proto_weights=pweights,
            mean_distance=dists,
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

    def test_ccnv_error_contracts_in_binning_rebinning_and_cohort(self) -> None:
        with self.assertRaises(InvalidInputError) as ctx1:
            bin_intervals(0, 5000)
        self.assertIsInstance(ctx1.exception, CcnvError)

        with self.assertRaises(InvalidInputError):
            rebin_counts(
                np.array([0], dtype=np.int64),
                np.array([5000], dtype=np.int64),
                np.array([10.0]),
                np.array([0], dtype=np.int64),
                np.array([10_000], dtype=np.int64),
            )

        bufs = _make_sample_buffers(
            {"chr1": 10_000}, 5000, np.array([10, 20], dtype=np.uint32)
        )
        with self.assertRaises(InvalidInputError):
            rebin_sample_buffers(bufs, 0)

        cohort = validate_cohort((bufs.to_table(),))
        with self.assertRaises(InvalidInputError):
            rebin_cohort(cohort, 0)


if __name__ == "__main__":
    unittest.main()
