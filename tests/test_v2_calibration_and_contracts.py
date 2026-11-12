"""End-to-end calibration test suite and structural contract verification for ccnv v2."""

from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pysam
from scipy.stats import norm

from ccnv._core import FRAG_BAND_EDGES
from ccnv.extract import run as run_extract
from ccnv.features.prenatal_origin import (
    PrenatalLoDMetrics,
    annotate_prenatal_origin_and_lod,
)
from ccnv.models import (
    ExtractConfig,
    PredictConfig,
    PredictionOutputs,
    ReferenceConfig,
)
from ccnv.models.results import SegmentResult
from ccnv.predict import run as run_predict
from ccnv.reference import run as run_reference
from ccnv.stats.fragmentomics import FRAG_MAX_LEN, FRAG_MIN_LEN

CONTIGS_CALIBRATION: tuple[tuple[str, int], ...] = (
    ("chr1", 100_000),
    ("chr2", 100_000),
    ("chr3", 100_000),
    ("chr4", 100_000),
    ("chr10", 100_000),
)
CALIBRATION_BIN_SIZE = 5_000


class TestV2CalibrationAndContracts(unittest.TestCase):
    """End-to-end mathematical calibration and structural codebase contracts."""

    _tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    pred_bam: PredictionOutputs | None = None
    pred_cram: PredictionOutputs | None = None
    qc_path: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Build synthetic reference cohort and target BAM/CRAM once for calibration tests."""
        cls._tmp_dir = tempfile.TemporaryDirectory()
        root = Path(cls._tmp_dir.name)

        fasta = root / "reference.fa"
        with fasta.open("w", encoding="ascii") as handle:
            for name, length in CONTIGS_CALIBRATION:
                handle.write(f">{name}\n{'A' * length}\n")
        pysam.faidx(str(fasta))

        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": n, "LN": l} for n, l in CONTIGS_CALIBRATION],
        }

        # 1. Reference cohort: 8 control samples at ~1.0x depth (50 reads per 5 kb bin)
        sample_paths: list[Path] = []
        for s_idx in range(8):
            bam_path = root / f"ctrl_{s_idx}.bam"
            with pysam.AlignmentFile(str(bam_path), "wb", header=header) as out_bam:
                read_id = 0
                for ref_id, (_name, length) in enumerate(CONTIGS_CALIBRATION):
                    n_bins = length // CALIBRATION_BIN_SIZE
                    for b_idx in range(n_bins):
                        start = b_idx * CALIBRATION_BIN_SIZE
                        n_reads = 50 + (s_idx % 3) - 1
                        for r_offset in range(n_reads):
                            read = pysam.AlignedSegment()
                            read.query_name = f"r_{read_id}"
                            read.query_sequence = "A" * 50
                            read.flag = 0
                            read.reference_id = ref_id
                            read.reference_start = min(
                                start + r_offset * 90, length - 50
                            )
                            read.mapping_quality = 60
                            read.cigar = ((0, 50),)
                            read.query_qualities = pysam.qualitystring_to_array(
                                "I" * 50
                            )
                            out_bam.write(read)
                            read_id += 1
            pysam.index(str(bam_path))
            pfx = root / f"ctrl_{s_idx}"
            run_extract(
                ExtractConfig(
                    bam_path,
                    pfx,
                    CALIBRATION_BIN_SIZE,
                    all_contigs=True,
                )
            )
            sample_paths.append(root / f"ctrl_{s_idx}.ccnv")

        ref_pfx = root / "reference"
        run_reference(
            ReferenceConfig(
                tuple(sample_paths),
                ref_pfx,
                bin_size=CALIBRATION_BIN_SIZE,
                pca_components=2,
                nearest_bins=10,
                min_cohort_size=8,
                all_contigs=True,
            )
        )
        reference_parquet = root / "reference.ccnv"

        # 2. Low-depth target sample (~0.15x depth, baseline 10 reads per 5 kb bin)
        # chr2: bins 5 to 15 [25000, 75000) count = 0 (CN=0 homozygous deletion)
        # chr10: bins 5 to 15 [25000, 75000) count = 3 (CN=1 focal single-copy deletion)
        # chr1, chr3, chr4: neutral background (10 reads per bin)
        target_bam = root / "target.bam"
        with pysam.AlignmentFile(str(target_bam), "wb", header=header) as out_bam:
            read_id = 0
            for ref_id, (name, length) in enumerate(CONTIGS_CALIBRATION):
                n_bins = length // CALIBRATION_BIN_SIZE
                for b_idx in range(n_bins):
                    start = b_idx * CALIBRATION_BIN_SIZE
                    if name == "chr2" and 5 <= b_idx < 15:
                        n_reads = 0
                    elif name == "chr10" and 5 <= b_idx < 15:
                        n_reads = 3
                    else:
                        n_reads = 10
                    for r_offset in range(n_reads):
                        read = pysam.AlignedSegment()
                        read.query_name = f"t_{read_id}"
                        read.query_sequence = "A" * 50
                        read.flag = 0
                        read.reference_id = ref_id
                        read.reference_start = min(start + r_offset * 100, length - 50)
                        read.mapping_quality = 60
                        read.cigar = ((0, 50),)
                        read.query_qualities = pysam.qualitystring_to_array("I" * 50)
                        out_bam.write(read)
                        read_id += 1
        pysam.index(str(target_bam))

        target_cram = root / "target.cram"
        with (
            pysam.AlignmentFile(str(target_bam), "rb") as in_bam,
            pysam.AlignmentFile(
                str(target_cram),
                "wc",
                header=in_bam.header,
                reference_filename=str(fasta),
            ) as out_cram,
        ):
            for read in in_bam:
                out_cram.write(read)
        pysam.index(str(target_cram))

        cls.qc_path = root / "pred_bam_qc.json"
        cls.pred_bam = run_predict(
            PredictConfig(
                target_bam,
                reference_parquet,
                root / "pred_bam",
                fasta=fasta,
                penalty=0.05,
                ratio_threshold=0.2,
                min_reference_bins=3,
                plot=False,
            )
        )
        cls.pred_cram = run_predict(
            PredictConfig(
                target_cram,
                reference_parquet,
                root / "pred_cram",
                fasta=fasta,
                penalty=0.05,
                ratio_threshold=0.2,
                min_reference_bins=3,
                plot=False,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary test fixtures."""
        if cls._tmp_dir is not None:
            cls._tmp_dir.cleanup()

    def test_bam_vs_cram_end_to_end_mathematical_parity(self) -> None:
        """Verify exact mathematical parity of segment calls, copy numbers, and bin ratios."""
        self.assertIsNotNone(self.pred_bam)
        self.assertIsNotNone(self.pred_cram)
        if self.pred_bam is None or self.pred_cram is None:
            self.fail("Prediction outputs not initialized")

        bam_segs = self.pred_bam.segments
        cram_segs = self.pred_cram.segments
        self.assertEqual(len(bam_segs), len(cram_segs))

        for b_seg, c_seg in zip(bam_segs, cram_segs):
            self.assertEqual(b_seg.contig, c_seg.contig)
            self.assertEqual(b_seg.start, c_seg.start)
            self.assertEqual(b_seg.end, c_seg.end)
            self.assertEqual(b_seg.call, c_seg.call)
            self.assertEqual(b_seg.copy_number, c_seg.copy_number)
            self.assertAlmostEqual(b_seg.ratio, c_seg.ratio, places=9)
            self.assertAlmostEqual(b_seg.zscore, c_seg.zscore, places=9)

        self.assertEqual(len(self.pred_bam.bins), len(self.pred_cram.bins))
        for b_bin, c_bin in zip(self.pred_bam.bins, self.pred_cram.bins):
            np.testing.assert_allclose(b_bin.ratios, c_bin.ratios, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(
                b_bin.zscores, c_bin.zscores, rtol=1e-8, atol=1e-8
            )

    def test_low_depth_0_1x_cn0_homozygous_and_focal_50kb_deletion_recovery(
        self,
    ) -> None:
        """Verify 0.1x depth recovers CN=0 homozygous deletion and interior focal loss with 0 FPR."""
        self.assertIsNotNone(self.pred_bam)
        if self.pred_bam is None:
            self.fail("pred_bam is None")

        segs = self.pred_bam.segments
        chr2_del = [
            s
            for s in segs
            if s.contig == "chr2" and s.start == 25_000 and s.end == 75_000
        ]
        self.assertEqual(len(chr2_del), 1)
        self.assertEqual(chr2_del[0].call, "loss")
        self.assertEqual(chr2_del[0].copy_number, 0)
        self.assertLess(chr2_del[0].ratio, -3.0)
        self.assertLess(chr2_del[0].zscore, -6.0)

        chr10_del = [
            s
            for s in segs
            if s.contig == "chr10" and s.start == 25_000 and s.end == 75_000
        ]
        self.assertEqual(len(chr10_del), 1)
        self.assertEqual(chr10_del[0].call, "loss")
        self.assertEqual(chr10_del[0].copy_number, 1)
        self.assertLess(chr10_del[0].ratio, -0.8)
        self.assertLess(chr10_del[0].zscore, -3.0)

        neutral_calls = [
            s
            for s in segs
            if s.contig in {"chr1", "chr3", "chr4"}
            and s.call not in {None, "NEUTRAL", "neutral"}
        ]
        self.assertEqual(len(neutral_calls), 0)

        self.assertIsNotNone(self.qc_path)
        if self.qc_path is None:
            self.fail("qc_path is None")
        self.assertTrue(self.qc_path.is_file())
        qc_data = json.loads(self.qc_path.read_text(encoding="utf-8"))
        self.assertIn("read_metrics", qc_data)
        self.assertEqual(qc_data["cnv_summary"]["loss_calls"], 2)
        self.assertEqual(qc_data["cnv_summary"]["gain_calls"], 0)

    def test_prenatal_5state_origin_and_mosaicism_classification_recovery(self) -> None:
        """Verify 5-state origin recovery at low fetal fraction and analytical LoD metrics."""
        lengths = np.arange(FRAG_MIN_LEN, FRAG_MAX_LEN + 1, dtype=np.float64)
        p_f = norm.pdf(lengths, loc=143.0, scale=10.0)
        p_f[lengths < 155] *= 1.0 + 0.15 * np.cos(
            2.0 * np.pi * (lengths[lengths < 155] - 143.0) / 10.4
        )
        p_f = np.maximum(p_f, 0.0)
        p_f /= np.sum(p_f)

        p_m = norm.pdf(lengths, loc=166.0, scale=12.0)
        p_m /= np.sum(p_m)

        edges = np.asarray(FRAG_BAND_EDGES, dtype=np.int64)
        n_bands = len(edges) - 1
        p_f_24 = np.zeros(n_bands, dtype=np.float64)
        p_m_24 = np.zeros(n_bands, dtype=np.float64)
        for k in range(n_bands):
            s = int(edges[k] - FRAG_MIN_LEN)
            e = int(edges[k + 1] - FRAG_MIN_LEN)
            p_f_24[k] = np.sum(p_f[s:e])
            p_m_24[k] = np.sum(p_m[s:e])
        p_f_24 = np.clip(p_f_24, 1e-8, None)
        p_f_24 /= np.sum(p_f_24)
        p_m_24 = np.clip(p_m_24, 1e-8, None)
        p_m_24 /= np.sum(p_m_24)

        active_ff = 0.08
        active_ff_se = 0.005
        p0 = active_ff * p_f_24 + (1.0 - active_ff) * p_m_24

        n_bins = 400
        bin_contigs = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100, dtype=np.int16
        )
        contig_names = ["chr1", "chr2", "chr3", "chr21"]
        bin_starts = np.tile(np.arange(100, dtype=np.int64) * 20_000, 4)
        bin_ends = bin_starts + 20_000
        bin_usable = np.ones(n_bins, dtype=np.bool_)
        sample_bands = np.zeros((n_bins, 24), dtype=np.uint32)

        for b in range(n_bins):
            sample_bands[b] = np.round(500 * p0).astype(np.uint32)

        # 1. Chr21: Fetal Constitutional Trisomy 21 (R = 1 + 0.5 * 0.08 = 1.04)
        phi_fc = (1.5 * active_ff) / (1.0 - active_ff + 1.5 * active_ff)
        p_fc = phi_fc * p_f_24 + (1.0 - phi_fc) * p_m_24
        for b in range(300, 350):
            sample_bands[b] = np.round(520 * p_fc).astype(np.uint32)
        seg0 = SegmentResult(
            contig="chr21",
            start=0,
            end=1_000_000,
            ratio=math.log2(1.04),
            zscore=4.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # 2. Chr1: Maternal+Fetal Inherited Duplication (R = 1.50)
        for b in range(10, 60):
            sample_bands[b] = np.round(750 * p0).astype(np.uint32)
        seg1 = SegmentResult(
            contig="chr1",
            start=200_000,
            end=1_200_000,
            ratio=math.log2(1.50),
            zscore=12.0,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # 3. Chr2: 40% CPM gain
        r_cpm = 1.0 + 0.5 * 0.40 * active_ff
        phi_cpm = (1.0 + 0.5 * 0.40) * active_ff / r_cpm
        p_cpm = phi_cpm * p_f_24 + (1.0 - phi_cpm) * p_m_24
        for b in range(110, 160):
            sample_bands[b] = np.round(508 * p_cpm).astype(np.uint32)
        seg2 = SegmentResult(
            contig="chr2",
            start=200_000,
            end=1_200_000,
            ratio=math.log2(r_cpm),
            zscore=3.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        # 4. Chr3: 4.4% Maternal Somatic Mosaic gain
        r_mat = 1.0 + 0.5 * 0.044 * (1.0 - active_ff)
        phi_mat = active_ff / r_mat
        p_mat = phi_mat * p_f_24 + (1.0 - phi_mat) * p_m_24
        for b in range(210, 260):
            sample_bands[b] = np.round(510 * p_mat).astype(np.uint32)
        seg3 = SegmentResult(
            contig="chr3",
            start=200_000,
            end=1_200_000,
            ratio=math.log2(r_mat),
            zscore=3.5,
            weight=50.0,
            bin_count=50,
            call="gain",
        )

        annotated, lod = annotate_prenatal_origin_and_lod(
            segments=(seg0, seg1, seg2, seg3),
            sample_bands=sample_bands,
            bin_contigs=bin_contigs,
            contig_names=contig_names,
            bin_usable=bin_usable,
            p_fetal_24=p_f_24,
            mu_fetal_w=1.0,
            mu_maternal_w=1.0,
            active_ff=active_ff,
            active_ff_se=active_ff_se,
            sca_call="XX",
            sample_depth_per_mb=25_000.0,
            sys_var_per_mb=0.0001,
            bin_starts=bin_starts,
            bin_ends=bin_ends,
            p_maternal_24=p_m_24,
        )

        self.assertEqual(len(annotated), 4)
        self.assertEqual(annotated[0].origin_call, "FETAL_CONSTITUTIONAL")
        self.assertEqual(annotated[1].origin_call, "MATERNAL_AND_FETAL_INHERITED")
        self.assertEqual(annotated[2].origin_call, "PLACENTAL_MOSAIC")
        self.assertIsNotNone(annotated[2].mosaic_fraction)
        self.assertGreater(float(annotated[2].mosaic_fraction or 0.0), 0.25)
        self.assertLess(float(annotated[2].mosaic_fraction or 0.0), 0.55)
        self.assertEqual(annotated[3].origin_call, "MATERNAL_MOSAIC")

        self.assertIsInstance(lod, PrenatalLoDMetrics)
        self.assertGreater(lod.lod_ff_35mb, 0.0)
        self.assertLess(lod.lod_ff_35mb, lod.lod_ff_3mb)
        self.assertLess(lod.lod_ff_3mb, 0.50)

    def test_repository_line_count_invariants(self) -> None:
        """Enforce strict < 1,000 line length limit across all Python modules in src/ and tests/."""
        repo_root = Path(__file__).resolve().parent.parent
        long_files: list[tuple[str, int]] = []

        for base_dir in ("src", "tests"):
            for py_path in (repo_root / base_dir).glob("**/*.py"):
                line_count = len(py_path.read_text(encoding="utf-8").splitlines())
                if line_count >= 1_000:
                    long_files.append((str(py_path.relative_to(repo_root)), line_count))

        self.assertEqual(
            long_files,
            [],
            f"The following files exceed the 1,000-line boundary: {long_files}",
        )

    def test_ast_cleanliness_zero_any_type_ignore_bare_assert(self) -> None:
        """Verify zero typing.Any, zero bare assert statements, and zero # type: ignore in src/."""
        repo_root = Path(__file__).resolve().parent.parent
        any_violations: list[tuple[str, int]] = []
        assert_violations: list[tuple[str, int]] = []
        type_ignore_violations: list[tuple[str, int, str]] = []

        for py_path in (repo_root / "src").glob("**/*.py"):
            rel_path = str(py_path.relative_to(repo_root))
            text = py_path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), start=1):
                if "# type: ignore" in line:
                    type_ignore_violations.append((rel_path, idx, line.strip()))

            tree = ast.parse(text, filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    assert_violations.append((rel_path, node.lineno))
                elif isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        if alias.name == "Any":
                            any_violations.append((rel_path, node.lineno))

        self.assertEqual(
            any_violations,
            [],
            f"Found forbidden typing.Any imports in src/: {any_violations}",
        )
        self.assertEqual(
            assert_violations,
            [],
            f"Found forbidden bare assert statements in src/: {assert_violations}",
        )
        self.assertEqual(
            type_ignore_violations,
            [],
            f"Found forbidden # type: ignore comments in src/: {type_ignore_violations}",
        )

    def test_ultra_low_depth_cn0_recovery_with_default_mbic_penalty(self) -> None:
        """Verify true ~0.1x depth (3 reads/bin) recovers CN=0 under default penalty=1.0 mBIC."""
        self.assertIsNotNone(self._tmp_dir)
        if self._tmp_dir is None:
            self.fail("_tmp_dir is None")
        root = Path(self._tmp_dir.name)
        reference_parquet = root / "reference.ccnv"

        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": n, "LN": l} for n, l in CONTIGS_CALIBRATION],
        }
        ultra_bam = root / "ultra_low.bam"
        with pysam.AlignmentFile(str(ultra_bam), "wb", header=header) as out_bam:
            read_id = 0
            for ref_id, (name, length) in enumerate(CONTIGS_CALIBRATION):
                n_bins = length // CALIBRATION_BIN_SIZE
                for b_idx in range(n_bins):
                    start = b_idx * CALIBRATION_BIN_SIZE
                    n_reads = 0 if (name == "chr2" and 5 <= b_idx < 15) else 3
                    for r_offset in range(n_reads):
                        read = pysam.AlignedSegment()
                        read.query_name = f"u_{read_id}"
                        read.query_sequence = "A" * 50
                        read.flag = 0
                        read.reference_id = ref_id
                        read.reference_start = min(start + r_offset * 200, length - 50)
                        read.mapping_quality = 60
                        read.cigar = ((0, 50),)
                        read.query_qualities = pysam.qualitystring_to_array("I" * 50)
                        out_bam.write(read)
                        read_id += 1

        pred_ultra = run_predict(
            PredictConfig(
                ultra_bam,
                reference_parquet,
                root / "pred_ultra",
                min_reference_bins=3,
                plot=False,
                vcf=False,
            )
        )
        chr2_del = [
            s
            for s in pred_ultra.segments
            if s.contig == "chr2" and s.start == 25_000 and s.end == 75_000
        ]
        self.assertEqual(len(chr2_del), 1)
        self.assertEqual(chr2_del[0].call, "loss")
        self.assertEqual(chr2_del[0].copy_number, 0)
        self.assertIsNotNone(chr2_del[0].log_bf)
        self.assertGreater(float(chr2_del[0].log_bf or 0.0), 20.0)

    def test_zero_pysam_or_pybigwig_imports_in_src_ccnv(self) -> None:
        """Verify zero pysam or pyBigWig runtime imports exist in src/ccnv/."""
        repo_root = Path(__file__).resolve().parent.parent
        forbidden_imports: list[tuple[str, int, str]] = []

        for py_path in (repo_root / "src" / "ccnv").glob("**/*.py"):
            rel_path = str(py_path.relative_to(repo_root))
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if "pysam" in mod or "pyBigWig" in mod or "pybigwig" in mod.lower():
                        forbidden_imports.append((rel_path, node.lineno, mod))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in {"pysam", "pyBigWig", "pybigwig"}:
                            forbidden_imports.append(
                                (rel_path, node.lineno, alias.name)
                            )

        self.assertEqual(
            forbidden_imports,
            [],
            f"Found forbidden pysam/pyBigWig imports in src/ccnv/: {forbidden_imports}",
        )


if __name__ == "__main__":
    unittest.main()
