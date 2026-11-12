from __future__ import annotations

import csv
import gzip
import json
import math
import tempfile
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import published_fixtures as pf
import pyarrow.parquet as pq
import pysam
import yaml
from helpers import make_sample_table, write_synthetic_sample
from numpy.typing import NDArray
from typer.testing import CliRunner

import ccnv.predict as P
from ccnv.cli import app
from ccnv.extract import run as extract
from ccnv.features.sex import infer_single_sample_sex
from ccnv.io.parquet import read_reference_parquet, read_sample_buffers
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BinResult,
    ExtractConfig,
    PredictConfig,
    PredictionOutputs,
    ReferenceConfig,
    ReferenceMetadata,
    SampleBuffers,
    SamplesheetEntry,
    SegmentResult,
    parquet_path,
    validate_reference_table,
)
from ccnv.models.schemas import SampleMetadata, decode_metadata, reconstruct_sample_grid
from ccnv.predict import run as predict
from ccnv.reference import run as reference
from ccnv.report import BINS_BED_HEADER, CALLS_BED_HEADER, SEGMENTS_BED_HEADER
from ccnv.stats.segmentation import segment_contig

CONTIGS = (("chr2", 1_200), ("chr10", 1_150), ("linkage-X", 1_080), ("plasmid-Y", 930))
CONTRACT_BIN_SIZE = 100
FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"
BENCHMARK_BIN_SIZE = 100_000
CONTIGS_CALIBRATION = tuple(
    (c, 100_000) for c in ("chr1", "chr2", "chr3", "chr4", "chr10")
)
CALIBRATION_BIN_SIZE = 5_000


def _zero_disp_branch_model(
    contigs: tuple[str, ...], starts: list[int], counts: list[float]
) -> P.BranchModel:
    n = len(contigs)
    pids = np.zeros((n, 4), dtype=np.uint16)
    pweights = np.zeros((n, 4), dtype=np.float16)
    pweights[:, 0] = np.float16(1.0)
    st, z64 = np.asarray(starts, dtype=np.int64), np.zeros(n, dtype=np.float64)
    c_arr, m_arr = np.asarray(counts, dtype=np.float64), np.zeros(n, dtype=np.bool_)
    pm, pl = np.full(n, 0.25), np.zeros((1, n), dtype=np.float32)
    md = np.ones(n, dtype=np.float32)
    return P.BranchModel(
        BRANCH_AUTOSOMAL,
        contigs,
        st,
        st + 100,
        c_arr,
        m_arr,
        pm,
        pl,
        pids,
        pweights,
        md,
        z64,
        z64,
    )


def _find_segs(
    segs: Sequence[SegmentResult], contig: str, start: int, end: int
) -> list[SegmentResult]:
    return [s for s in segs if s.contig == contig and s.start == start and s.end == end]


def _control_counts(sample_index: int) -> dict[str, NDArray[np.int64]]:
    result: dict[str, NDArray[np.int64]] = {}
    for contig_index, (contig, length) in enumerate(CONTIGS):
        bins = (length + CONTRACT_BIN_SIZE - 1) // CONTRACT_BIN_SIZE
        idx = np.arange(bins)
        vals = 20 + (idx * 3 + contig_index * 2 + sample_index) % 7 + (sample_index % 3)
        result[contig] = vals.astype(np.int64)
    return result


def _write_fasta(path: Path, contigs: tuple[tuple[str, int], ...] = CONTIGS) -> None:
    with path.open("w", encoding="ascii") as handle:
        for name, length in contigs:
            handle.write(f">{name}\n{'A' * length}\n")
    pysam.faidx(str(path))


def _write_se_bam(
    path: Path,
    contigs: tuple[tuple[str, int], ...],
    bin_size: int,
    read_count_fn: Callable[[str, int], int],
    *,
    read_len: int = 50,
    prefix_tag: str = "r",
    step: int = 1,
    index: bool = True,
) -> None:
    sq = [{"SN": name, "LN": length} for name, length in contigs]
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": sq}
    seq = "A" * read_len
    quals = pysam.qualitystring_to_array("I" * read_len)
    cigar = ((0, read_len),)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out_bam:
        read_id = 0
        for ref_id, (name, length) in enumerate(contigs):
            n_bins = (length + bin_size - 1) // bin_size
            for b_idx in range(n_bins):
                start = b_idx * bin_size
                for offset in range(read_count_fn(name, b_idx)):
                    r = pysam.AlignedSegment()
                    r.query_name, r.query_sequence, r.flag = (
                        f"{prefix_tag}_{read_id}",
                        seq,
                        0,
                    )
                    r.reference_id = ref_id
                    r.reference_start = min(start + offset * step, length - read_len)
                    r.mapping_quality, r.cigar, r.query_qualities = 60, cigar, quals
                    out_bam.write(r)
                    read_id += 1
    if index:
        pysam.index(str(path))


def _write_alignment(path: Path, counts: dict[str, NDArray[np.int64]]) -> None:
    _write_se_bam(
        path, CONTIGS, CONTRACT_BIN_SIZE, lambda c, b: int(counts[c][b]), read_len=10
    )


def _bam_to_cram(bam_path: Path, cram_path: Path, fasta: Path) -> None:
    ref_fn = str(fasta)
    with (
        pysam.AlignmentFile(str(bam_path), "rb") as bam,
        pysam.AlignmentFile(
            str(cram_path), "wc", header=bam.header, reference_filename=ref_fn
        ) as cram,
    ):
        for read in bam:
            cram.write(read)
    pysam.index(str(cram_path))


def _write_synthetic_bam(bam_path: Path) -> Path:
    sq = [
        {"SN": c, "LN": ln}
        for c, ln in (("chr1", 1200), ("chr2", 1200), ("chrX", 800), ("chrY", 800))
    ]
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": sq}
    rng = np.random.default_rng(7)
    quals = pysam.qualitystring_to_array("I" * 50)
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as outf:
        read_id = 0
        for tid, length in enumerate([1200, 1200, 800]):
            for pos in range(10, length - 200, 15):
                tlen = int(rng.integers(150, 180))
                a = pysam.AlignedSegment(outf.header)
                a.query_name, a.query_sequence, a.flag = f"r{read_id}", "A" * 50, 99
                a.reference_id, a.reference_start, a.mapping_quality = tid, pos, 60
                a.cigar, a.next_reference_id = ((0, 50),), tid
                a.next_reference_start, a.template_length = pos + tlen - 50, tlen
                a.query_qualities = quals
                outf.write(a)
                read_id += 1
    pysam.index(str(bam_path))
    return bam_path


class TestInstalledPipelineContract(unittest.TestCase):
    def test_bam_and_cram_to_gain_loss_calls_on_nonhuman_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = root / "genome.fa"
            _write_fasta(fasta)

            sample_paths: list[Path] = []
            for s_idx in range(8):
                bam = root / f"control-{s_idx}.bam"
                _write_alignment(bam, _control_counts(s_idx))
                aln = bam
                if s_idx == 0:
                    aln = root / "control-0.cram"
                    _bam_to_cram(bam, aln, fasta)
                pfx = root / f"control-{s_idx}"
                fa_arg = fasta if aln.suffix == ".cram" else None
                extract(
                    ExtractConfig(
                        aln, pfx, CONTRACT_BIN_SIZE, fasta=fa_arg, all_contigs=True
                    )
                )
                sample_paths.append(parquet_path(pfx))

            target_counts = _control_counts(0)
            target_counts["chr10"][4:8] *= 2
            target_counts["linkage-X"][4:8] = np.maximum(
                1, target_counts["linkage-X"][4:8] // 4
            )
            target_bam = root / "target.bam"
            target_pfx = root / "target"
            target_sample = parquet_path(target_pfx)
            _write_alignment(target_bam, target_counts)
            extract(
                ExtractConfig(
                    target_bam, target_pfx, CONTRACT_BIN_SIZE, all_contigs=True
                )
            )

            ref_pfx = root / "reference"
            ref_path = parquet_path(ref_pfx)
            entries = tuple(SamplesheetEntry(filepath=p) for p in sample_paths)
            reference(
                ReferenceConfig(
                    prefix=ref_pfx, entries=entries, pca_components=2, min_cohort_size=8
                )
            )
            pred_cfg = PredictConfig(
                target_sample,
                ref_path,
                root / "prediction",
                penalty=0.05,
                ratio_threshold=0.2,
                min_reference_bins=3,
                plot=True,
            )
            outputs = predict(pred_cfg)

            sample = pq.read_table(target_sample)
            ref_identity = decode_metadata(sample.schema.metadata)["reference_identity"]
            if not isinstance(ref_identity, dict):
                self.fail("expected reference_identity dict")
            self.assertEqual(ref_identity["contig_order"], [n for n, _ in CONTIGS])
            s_meta = SampleMetadata.from_table(sample)
            codes, names, _starts, ends = reconstruct_sample_grid(s_meta)
            self.assertIn("chr10", names)
            self.assertEqual(int(ends[codes == names.index("plasmid-Y")][-1]), 930)

            calls = {(item.contig, item.call) for item in outputs.segments if item.call}
            evidence = [
                (
                    r.contigs[i],
                    int(r.starts[i]),
                    float(r.ratios[i]),
                    float(r.zscores[i]),
                    bool(r.masked[i]),
                )
                for r in outputs.bins
                for i in range(len(r.contigs))
            ]
            self.assertIn(("chr10", "gain"), calls, evidence)
            self.assertIn(("linkage-X", "loss"), calls, evidence)
            self.assertTrue((root / "reference_qc.json").is_file())
            for bed_p in (outputs.bins_bed, outputs.segments_bed, outputs.calls_bed):
                self.assertTrue(bed_p.is_file())
                self.assertTrue(Path(f"{bed_p}.tbi").is_file())
            self.assertIsNone(outputs.bins_vcf)
            self.assertFalse((root / "prediction_bins.vcf.gz").exists())
            self.assertIsNotNone(outputs.segments_vcf)
            self.assertIsNotNone(outputs.calls_vcf)
            if outputs.segments_vcf is None or outputs.calls_vcf is None:
                self.fail("expected segments_vcf and calls_vcf")
            for vcf_p in (outputs.segments_vcf, outputs.calls_vcf):
                self.assertTrue(vcf_p.is_file())
                self.assertTrue(Path(f"{vcf_p}.tbi").is_file())
            self.assertFalse((root / "prediction_bins.ccnv").exists())
            self.assertFalse((root / "prediction_segments.ccnv").exists())
            self.assertTrue(
                outputs.plot_html is not None and outputs.plot_html.is_file()
            )

            with gzip.open(outputs.bins_bed, "rt") as h:
                bins_lines = h.read().splitlines()
            with gzip.open(outputs.segments_bed, "rt") as h:
                segments_header = h.read().splitlines()[0].split("\t")
            with gzip.open(outputs.calls_bed, "rt") as h:
                calls_header = h.read().splitlines()[0].split("\t")
            self.assertEqual(tuple(bins_lines[0].split("\t")), BINS_BED_HEADER)
            self.assertEqual(tuple(segments_header), SEGMENTS_BED_HEADER)
            self.assertEqual(tuple(calls_header), CALLS_BED_HEADER)
            for line in bins_lines[1:]:
                contig = line.split("\t")[0]
                self.assertFalse(contig.isdigit(), contig)
            for segment in outputs.segments:
                self.assertLessEqual(segment.end, dict(CONTIGS)[segment.contig])


class TestBenchmarkNumericAgreement(unittest.TestCase):
    def test_synthetic_segment_means_and_calls_are_concordant(self) -> None:
        bins_by_contig: dict[str, list[tuple[float, float, float]]] = {}
        with (FIXTURES / "synthetic" / "normalized_bins.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                bins_by_contig.setdefault(row["chr"], []).append(
                    (float(row["ratio"]), float(row["zscore"]), float(row["weight"]))
                )
        with (FIXTURES / "synthetic" / "expected_segments.tsv").open() as handle:
            expected_segments = {
                r["chr"]: (int(r["start_bin"]), int(r["end_bin"]), float(r["ratio"]))
                for r in csv.DictReader(handle, delimiter="\t")
            }
        with (FIXTURES / "synthetic" / "calls.tsv").open() as handle:
            expected_calls = {
                (r["chr"], r["type"]) for r in csv.DictReader(handle, delimiter="\t")
            }

        observed_calls: set[tuple[str, str]] = set()
        absolute_errors: list[float] = []
        for contig, rows in bins_by_contig.items():
            ratios, zscores, weights = zip(*rows, strict=True)
            starts = np.arange(len(rows), dtype=np.int64) * BENCHMARK_BIN_SIZE
            ends = np.arange(1, len(rows) + 1, dtype=np.int64) * BENCHMARK_BIN_SIZE
            c_len = len(rows) * BENCHMARK_BIN_SIZE
            segments = segment_contig(
                contig=contig,
                starts=starts,
                ends=ends,
                ratios=ratios,
                weights=weights,
                contig_length=c_len,
                penalty=0.01,
            )
            self.assertEqual(len(segments), 1)
            exp_start, exp_end, exp_ratio = expected_segments[contig]
            bs = BENCHMARK_BIN_SIZE
            self.assertLessEqual(abs(segments[0].start - exp_start * bs), bs)
            self.assertLessEqual(abs(segments[0].end - exp_end * bs), bs)
            absolute_errors.append(abs(segments[0].ratio - exp_ratio))

            r_arr = np.asarray(ratios, dtype=np.float64)
            z_arr = np.asarray(zscores, dtype=np.float64)
            w_mask = np.asarray(weights, dtype=np.float64) > 0.0
            nz = np.abs(r_arr) > 1e-9
            nz_prec = np.where(nz, np.square(z_arr / np.where(nz, r_arr, 1.0)), 100.0)
            prec_arr = np.where(w_mask, nz_prec, 0.0)
            n_refs = np.full(len(rows), 10, dtype=np.int64)
            m_arr = ~(np.isfinite(r_arr) & w_mask)
            cs = (contig,) * len(rows)
            bin_res = BinResult(
                "autosomal", cs, starts, ends, r_arr, z_arr, prec_arr, n_refs, m_arr
            )
            called = P.segment_and_call(
                (bin_res,),
                {contig: c_len},
                penalty=0.01,
                z_threshold=3.0,
                ratio_threshold=0.2,
            )
            for seg in called:
                if seg.call is not None:
                    observed_calls.add((contig, seg.call))

        self.assertLessEqual(float(np.max(absolute_errors)), 1e-12)
        self.assertEqual(observed_calls, expected_calls)

    def test_real_fixture_gain_is_concordant(self) -> None:
        bins: dict[str, list[tuple[int, int, float, float]]] = {}
        with (FIXTURES / "real" / "ID_bins.bed").open() as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                rv = float(r["ratio"])
                tup = (int(r["start"]), int(r["end"]), rv, float(math.isfinite(rv)))
                bins.setdefault(r["chr"], []).append(tup)
        with (FIXTURES / "real" / "ID_aberrations.bed").open() as handle:
            expected = [
                (r["chr"], int(r["start"]), int(r["end"]), r["type"])
                for r in csv.DictReader(handle, delimiter="\t")
            ]

        observed: list[tuple[str, int, int, str]] = []
        for contig, rows in bins.items():
            c0, c1, c2, c3 = zip(*rows, strict=True)
            segs = segment_contig(
                contig=contig,
                starts=c0,
                ends=c1,
                ratios=c2,
                weights=c3,
                contig_length=int(c1[-1]),
                penalty=0.05,
            )
            for seg in segs:
                if math.isfinite(seg.ratio) and abs(seg.ratio) > 0.05:
                    kind = "gain" if seg.ratio > 0.0 else "loss"
                    observed.append((contig, seg.start, seg.end, kind))

        self.assertEqual(len(expected), 1)
        ec, es, ee, et = expected[0]
        bs = BENCHMARK_BIN_SIZE
        matches = [
            x
            for x in observed
            if (x[0], x[3]) == (ec, et)
            and abs(x[1] - es) <= bs
            and abs(x[2] - ee) <= bs
        ]
        self.assertEqual(len(matches) / len(expected), 1.0)
        self.assertGreaterEqual(len(matches) / len(observed), 0.5)


class TestReferencePublishedCrams(unittest.TestCase):
    def test_cli_builds_reference_from_two_published_crams(self) -> None:
        fasta = pf.ensure_published_fasta()
        crams = (
            pf.ensure_published_file(pf.PUBLISHED_CRAM, pf.PUBLISHED_CRAM_URL),
            pf.ensure_published_file(pf.PUBLISHED_CRAM2, pf.PUBLISHED_CRAM2_URL),
        )
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_paths: list[Path] = []
            bs_str = str(pf.PUBLISHED_BIN_SIZE)
            for index, cram in enumerate(crams, start=1):
                pfx = root / f"sample{index}"
                ext_args = ["extract", str(cram), "--fasta", str(fasta)]
                ext_args += ["--prefix", str(pfx), "--bin-size", bs_str]
                res_ext = runner.invoke(app, ext_args)
                self.assertEqual(res_ext.exit_code, 0, res_ext.output)
                sample_paths.append(parquet_path(pfx))

            samples = [read_sample_buffers(path) for path in sample_paths]
            self.assertEqual(samples[0].num_rows, samples[1].num_rows)
            self.assertEqual(samples[0].meta.contig_order, ("chr21",))

            sheet_path = root / "samplesheet.yaml"
            sheet_txt = yaml.safe_dump([{"filepath": str(p)} for p in sample_paths])
            sheet_path.write_text(sheet_txt, encoding="utf-8")
            ref_pfx = root / "reference"
            ref_args = ["reference", "--samplesheet", str(sheet_path)]
            ref_args += ["--prefix", str(ref_pfx), "--bin-size", bs_str]
            ref_args += ["--min-cohort-size", "2", "--pca-components", "1"]
            result = runner.invoke(app, ref_args)
            self.assertEqual(result.exit_code, 0, result.output)

            table = pq.read_table(parquet_path(ref_pfx))
            validate_reference_table(table)
            metadata = decode_metadata(table.schema.metadata)
            model = metadata["model"]
            sample_identities = metadata["sample_identities"]
            if not isinstance(model, dict) or not isinstance(sample_identities, list):
                self.fail("expected model dict and sample_identities list")
            self.assertEqual(table.num_rows, samples[0].num_rows)
            self.assertEqual(set(table.column("contig").to_pylist()), {"chr21"})
            self.assertEqual(
                set(table.column("branch").to_pylist()), {BRANCH_AUTOSOMAL}
            )
            self.assertEqual(table.column("end").to_pylist()[-1], pf.CHR21_LENGTH)
            self.assertEqual(metadata["contig_lengths"], {"chr21": pf.CHR21_LENGTH})
            self.assertEqual(model["bin_size"], pf.PUBLISHED_BIN_SIZE)
            self.assertEqual(model["branches"], [BRANCH_AUTOSOMAL])
            self.assertEqual(model["autosomal_contigs"], ["chr21"])
            self.assertEqual(len(sample_identities), 2)
            unmasked = sum(1 for masked in table.column("masked") if not masked)
            self.assertGreater(unmasked, 0)


class TestPredictPublishedCrams(unittest.TestCase):
    def test_cli_predicts_published_sample_against_two_sample_reference(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_a, sample_b = pf.extract_published_samples(root)
            ref_pfx = root / "reference"
            entries = (SamplesheetEntry(sample_a), SamplesheetEntry(sample_b))
            ref_cfg = ReferenceConfig(
                prefix=ref_pfx, entries=entries, pca_components=1, min_cohort_size=2
            )
            reference(ref_cfg)
            ref_pq = str(parquet_path(ref_pfx))
            pred_pfx = root / "prediction"
            pred_args = ["predict", str(sample_a), ref_pq, "--prefix", str(pred_pfx)]
            result = runner.invoke(app, pred_args)
            self.assertEqual(result.exit_code, 0, result.output)

            for stem in (
                "_bins.bed",
                "_segments.bed",
                "_calls.bed",
                "_segments.vcf",
                "_calls.vcf",
            ):
                self.assertTrue(Path(f"{pred_pfx}{stem}.gz").is_file())
                self.assertTrue(Path(f"{pred_pfx}{stem}.gz.tbi").is_file())
            for sfx in ("_plot.html", "_qc.json"):
                self.assertTrue(Path(f"{pred_pfx}{sfx}").is_file())
            for sfx in ("_bins.vcf.gz", "_bins.vcf.gz.tbi"):
                self.assertFalse(Path(f"{pred_pfx}{sfx}").exists())
            for sfx in ("_bins.ccnv", "_segments.ccnv"):
                self.assertFalse(Path(f"{pred_pfx}{sfx}").exists())

            qc_data = json.loads(
                Path(f"{pred_pfx}_qc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(qc_data["mode"], "germline")
            self.assertEqual(qc_data["schema_version"], "2.0.0")

            with gzip.open(f"{pred_pfx}_bins.bed.gz", "rt") as h:
                bin_lines = h.read().splitlines()
            self.assertEqual(tuple(bin_lines[0].split("\t")), BINS_BED_HEADER)
            self.assertEqual(len(bin_lines) - 1, 468)
            self.assertEqual(bin_lines[1].split("\t")[0], "chr21")
            self.assertEqual(int(bin_lines[1].split("\t")[1]), 0)
            self.assertEqual(int(bin_lines[-1].split("\t")[2]), pf.CHR21_LENGTH)

            with gzip.open(f"{pred_pfx}_segments.bed.gz", "rt") as h:
                seg_hdr = tuple(h.read().splitlines()[0].split("\t"))
            self.assertEqual(seg_hdr, SEGMENTS_BED_HEADER)
            with gzip.open(f"{pred_pfx}_calls.bed.gz", "rt") as h:
                call_hdr = tuple(h.read().splitlines()[0].split("\t"))
            self.assertEqual(call_hdr, CALLS_BED_HEADER)
            self.assertGreater(Path(f"{pred_pfx}_plot.html").stat().st_size, 0)

            opt_pfx = root / "prediction_with_bins"
            opt_args = ["predict", str(sample_a), ref_pq, "--prefix", str(opt_pfx)]
            opt_args += ["--bins-vcf", "--no-plot"]
            res_bins = runner.invoke(app, opt_args)
            self.assertEqual(res_bins.exit_code, 0, res_bins.output)
            opt_bins_vcf = Path(f"{opt_pfx}_bins.vcf.gz")
            self.assertTrue(opt_bins_vcf.is_file())
            self.assertTrue(Path(f"{opt_bins_vcf}.tbi").is_file())
            with pysam.TabixFile(str(opt_bins_vcf)) as tb:
                self.assertEqual(len(list(tb.fetch("chr21", 0, pf.CHR21_LENGTH))), 468)

    def test_normalize_branch_zero_bio_dispersion(self) -> None:
        model = _zero_disp_branch_model(
            ("chr1", "chr1", "chr2", "chr2"), [0, 100, 0, 100], [20.0, 30.0, 40.0, 50.0]
        )
        cfg = PredictConfig(Path("s"), Path("r"), Path("o"), min_reference_bins=2)
        result = P.normalize_branch(model, cfg)
        self.assertEqual(result.ratios.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.zscores)))

    def test_normalize_branch_flat_diploid_with_outlier_bin(self) -> None:
        model = _zero_disp_branch_model(
            ("chr1",) * 3 + ("chr2",) * 3,
            [0, 100, 200, 0, 100, 200],
            [20.0, 20.0, 20.0, 20.0, 20.0, 80.0],
        )
        cfg = PredictConfig(Path("s"), Path("r"), Path("o"), min_reference_bins=2)
        result = P.normalize_branch(model, cfg)
        self.assertEqual(result.ratios.shape, (6,))
        self.assertTrue(np.all(np.isfinite(result.zscores)))
        self.assertGreater(result.zscores[5], 2.0)

    def test_resolve_sample_sex_inferences_and_validations(self) -> None:
        c_lens = {"chr1": 100, "chrY": 100}
        sample = SampleBuffers.from_table(
            make_sample_table(
                [90, 10],
                contigs=["chr1", "chrY"],
                starts=[0, 0],
                ends=[100, 100],
                contig_lengths=c_lens,
                bin_size=100,
            )
        )
        sex_br = (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX)
        auto_only = ReferenceMetadata(100, c_lens, (BRANCH_AUTOSOMAL,))
        no_cut = ReferenceMetadata(100, c_lens, sex_br)
        low_cut = ReferenceMetadata(
            100, c_lens, sex_br, sex_signal_contig="chrY", sex_cutoff=0.05
        )
        high_cut = ReferenceMetadata(
            100, c_lens, sex_br, sex_signal_contig="chrY", sex_cutoff=0.15
        )

        self.assertEqual(infer_single_sample_sex(sample, auto_only, "female"), "female")
        self.assertEqual(infer_single_sample_sex(sample, auto_only, "male"), "male")
        self.assertIsNone(infer_single_sample_sex(sample, auto_only))
        with self.assertRaises(P.PredictError):
            infer_single_sample_sex(sample, no_cut)
        self.assertEqual(infer_single_sample_sex(sample, low_cut), "male")
        self.assertEqual(infer_single_sample_sex(sample, high_cut), "female")


class TestStreamingBamPipeline(unittest.TestCase):
    def test_end_to_end_reference_and_parallel_bam_predict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cohort_dir = tmp_path / "bam_cohort"
            rng = np.random.default_rng(11)
            contigs = ["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4
            starts = [0, 200, 400, 600, 800, 1000] * 2 + [0, 200, 400, 600] * 2
            ends = [s + 200 for s in starts]
            clens = {"chr1": 1200, "chr2": 1200, "chrX": 800, "chrY": 800}
            sample_paths: list[Path] = []
            for idx in range(6):
                male = idx % 2 == 1
                sex_c = ([15] * 4 if male else [30] * 4) + (
                    [12] * 4 if male else [0] * 4
                )
                counts = np.array([30 + idx] * 12 + sex_c, dtype=np.uint32)
                flens = [
                    rng.integers(150, 180, size=int(c)).tolist() if c > 0 else []
                    for c in counts
                ]
                p = cohort_dir / f"s_{idx}.ccnv"
                write_synthetic_sample(
                    p,
                    counts,
                    contigs=contigs,
                    starts=starts,
                    ends=ends,
                    contig_lengths=clens,
                    bin_size=200,
                    fragment_lengths=flens,
                    endpoint_counts=np.minimum(counts, 4),
                )
                sample_paths.append(p)

            entries = tuple(SamplesheetEntry(filepath=p) for p in sample_paths)
            ref_cfg = ReferenceConfig(
                prefix=tmp_path / "ref",
                entries=entries,
                bin_size=200,
                threads=2,
                min_cohort_size=4,
                min_sex_group_size=2,
            )
            ref_path = reference(ref_cfg)
            self.assertEqual(pq.ParquetFile(ref_path).num_row_groups, 3)

            _, ref_meta = read_reference_parquet(ref_path, branches=[BRANCH_AUTOSOMAL])
            bam_path = _write_synthetic_bam(tmp_path / "query.bam")
            extracted = P.extract_for_predict(
                bam_path, ref_meta, threads=2, collect_bands=False
            )
            self.assertIsNone(extracted.frag_bands)
            self.assertIsInstance(extracted.count, np.ndarray)
            self.assertEqual(extracted.count.shape, (20,))
            self.assertGreater(int(np.sum(extracted.count)), 0)

            err = AssertionError("must not materialize SAMPLE_SCHEMA table")
            with patch("ccnv.models.buffers.SampleBuffers.to_table", side_effect=err):
                direct = P.load_inputs(bam_path, ref_path, mode="germline", threads=2)
                self.assertIsInstance(direct.sample, SampleBuffers)
                self.assertIsNone(direct.sample.frag_bands)

                out_pfx = tmp_path / "pred" / "sample_out"
                cli_args = ["predict", str(bam_path), str(ref_path), "--prefix"]
                cli_args += [str(out_pfx), "-t", "2", "--no-bins-vcf", "--no-plot"]
                res = CliRunner().invoke(app, cli_args)
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertTrue(Path(f"{out_pfx}_bins.bed.gz").is_file())
            self.assertTrue(Path(f"{out_pfx}_segments.bed.gz").is_file())
            self.assertTrue(Path(f"{out_pfx}_qc.json").is_file())
            self.assertEqual(list((tmp_path / "pred").glob("*.ccnv")), [])


class TestEndToEndCalibration(unittest.TestCase):
    _tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    pred_bam: PredictionOutputs | None = None
    pred_cram: PredictionOutputs | None = None
    qc_path: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        root = Path(cls._tmp_dir.name)
        fasta = root / "reference.fa"
        _write_fasta(fasta, CONTIGS_CALIBRATION)

        sample_paths: list[Path] = []
        for s_idx in range(8):
            bam_path = root / f"ctrl_{s_idx}.bam"
            n_ctrl = 50 + (s_idx % 3) - 1
            _write_se_bam(
                bam_path,
                CONTIGS_CALIBRATION,
                CALIBRATION_BIN_SIZE,
                lambda _n, _b, c=n_ctrl: c,
                prefix_tag="r",
                step=90,
            )
            pfx = root / f"ctrl_{s_idx}"
            extract(
                ExtractConfig(bam_path, pfx, CALIBRATION_BIN_SIZE, all_contigs=True)
            )
            sample_paths.append(root / f"ctrl_{s_idx}.ccnv")

        entries = tuple(SamplesheetEntry(filepath=p) for p in sample_paths)
        ref_cfg = ReferenceConfig(
            prefix=root / "reference",
            entries=entries,
            bin_size=CALIBRATION_BIN_SIZE,
            pca_components=2,
            min_cohort_size=8,
            all_contigs=True,
        )
        reference(ref_cfg)
        ref_pq = root / "reference.ccnv"

        def _target_count(name: str, b_idx: int) -> int:
            if name == "chr2" and 5 <= b_idx < 15:
                return 0
            return 3 if (name == "chr10" and 5 <= b_idx < 15) else 10

        target_bam = root / "target.bam"
        _write_se_bam(
            target_bam,
            CONTIGS_CALIBRATION,
            CALIBRATION_BIN_SIZE,
            _target_count,
            prefix_tag="t",
            step=100,
        )
        target_cram = root / "target.cram"
        _bam_to_cram(target_bam, target_cram, fasta)

        cls.qc_path = root / "pred_bam_qc.json"
        for attr, aln, tag in (
            ("pred_bam", target_bam, "pred_bam"),
            ("pred_cram", target_cram, "pred_cram"),
        ):
            cfg = PredictConfig(
                aln,
                ref_pq,
                root / tag,
                fasta=fasta,
                penalty=0.05,
                ratio_threshold=0.2,
                min_reference_bins=3,
                plot=False,
            )
            setattr(cls, attr, predict(cfg))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmp_dir is not None:
            cls._tmp_dir.cleanup()

    def test_bam_vs_cram_end_to_end_mathematical_parity(self) -> None:
        self.assertIsNotNone(self.pred_bam)
        self.assertIsNotNone(self.pred_cram)
        if self.pred_bam is None or self.pred_cram is None:
            self.fail("Prediction outputs not initialized")

        bam_segs, cram_segs = self.pred_bam.segments, self.pred_cram.segments
        self.assertEqual(len(bam_segs), len(cram_segs))
        for b_seg, c_seg in zip(bam_segs, cram_segs, strict=True):
            self.assertEqual(b_seg.contig, c_seg.contig)
            self.assertEqual(b_seg.start, c_seg.start)
            self.assertEqual(b_seg.end, c_seg.end)
            self.assertEqual(b_seg.call, c_seg.call)
            self.assertEqual(b_seg.copy_number, c_seg.copy_number)
            self.assertAlmostEqual(b_seg.ratio, c_seg.ratio, places=9)
            self.assertAlmostEqual(b_seg.zscore, c_seg.zscore, places=9)

        self.assertEqual(len(self.pred_bam.bins), len(self.pred_cram.bins))
        for b_bin, c_bin in zip(self.pred_bam.bins, self.pred_cram.bins, strict=True):
            np.testing.assert_allclose(b_bin.ratios, c_bin.ratios, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(
                b_bin.zscores, c_bin.zscores, rtol=1e-8, atol=1e-8
            )

    def test_low_depth_0_1x_cn0_homozygous_and_focal_50kb_deletion_recovery(
        self,
    ) -> None:
        self.assertIsNotNone(self.pred_bam)
        if self.pred_bam is None:
            self.fail("pred_bam is None")

        segs = self.pred_bam.segments
        chr2_del = _find_segs(segs, "chr2", 25_000, 75_000)
        self.assertEqual(len(chr2_del), 1)
        self.assertEqual(chr2_del[0].call, "loss")
        self.assertEqual(chr2_del[0].copy_number, 0)
        self.assertLess(chr2_del[0].ratio, -3.0)
        self.assertLess(chr2_del[0].zscore, -6.0)

        chr10_del = _find_segs(segs, "chr10", 25_000, 75_000)
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

    def test_ultra_low_depth_cn0_recovery_with_default_mbic_penalty(self) -> None:
        self.assertIsNotNone(self._tmp_dir)
        if self._tmp_dir is None:
            self.fail("_tmp_dir is None")
        root = Path(self._tmp_dir.name)
        ultra_bam = root / "ultra_low.bam"
        _write_se_bam(
            ultra_bam,
            CONTIGS_CALIBRATION,
            CALIBRATION_BIN_SIZE,
            lambda name, b_idx: 0 if (name == "chr2" and 5 <= b_idx < 15) else 3,
            prefix_tag="u",
            step=200,
            index=False,
        )
        cfg = PredictConfig(
            ultra_bam,
            root / "reference.ccnv",
            root / "pred_ultra",
            min_reference_bins=3,
            plot=False,
            vcf=False,
        )
        pred_ultra = predict(cfg)
        chr2_del = _find_segs(pred_ultra.segments, "chr2", 25_000, 75_000)
        self.assertEqual(len(chr2_del), 1)
        self.assertEqual(chr2_del[0].call, "loss")
        self.assertEqual(chr2_del[0].copy_number, 0)
        self.assertIsNotNone(chr2_del[0].log_bf)
        self.assertGreater(float(chr2_del[0].log_bf or 0.0), 15.0)


if __name__ == "__main__":
    unittest.main()
