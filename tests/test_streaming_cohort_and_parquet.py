"""Tests for bounded-memory streaming cohort assembly, Prototype Graph, and Parquet v2 I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import pytest
from helpers import write_synthetic_sample
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.features.cohort import load_and_assemble_cohort
from ccnv.io.parquet import read_reference_parquet, write_reference_parquet
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    BinCompatibilityError,
    ReferenceConfig,
    ReferenceError,
    ReferenceMetadata,
    SampleEntry,
    build_reference_schema,
)
from ccnv.predict import build_branch_model, extract_for_predict
from ccnv.reference import run as build_reference
from ccnv.stats.neighbors import (
    evaluate_loco_prototype_baseline,
    fit_prototype_graph,
)


def _make_paired_cohort_files(
    tmp_path: Path,
    *,
    n_samples: int = 8,
    bin_size: int = 100,
) -> list[Path]:
    rng = np.random.default_rng(42)
    contigs = ["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4
    starts = [i * bin_size for i in range(6)] * 2 + [i * bin_size for i in range(4)] * 2
    ends = [s + bin_size for s in starts]
    contig_lengths = {
        "chr1": 6 * bin_size,
        "chr2": 6 * bin_size,
        "chrX": 4 * bin_size,
        "chrY": 4 * bin_size,
    }
    paths: list[Path] = []
    for idx in range(n_samples):
        is_male = idx % 2 == 1
        counts = np.array(
            [40 + idx] * 12
            + ([20] * 4 if is_male else [40] * 4)
            + ([18] * 4 if is_male else [0] * 4),
            dtype=np.uint32,
        )
        frag_lens = [
            rng.integers(145, 190, size=int(max(c, 1))).tolist() if c > 0 else []
            for c in counts
        ]
        endpoints = np.minimum(counts, 6).astype(np.uint32)
        p = tmp_path / f"sample_{idx:02d}.ccnv"
        write_synthetic_sample(
            p,
            counts,
            contigs=contigs,
            starts=starts,
            ends=ends,
            contig_lengths=contig_lengths,
            bin_size=bin_size,
            identity={"id": f"s_{idx:02d}"},
            fragment_lengths=frag_lens,
            endpoint_counts=endpoints,
        )
        paths.append(p)
    return paths


def _write_synthetic_bam(bam_path: Path) -> Path:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [
            {"SN": "chr1", "LN": 1200},
            {"SN": "chr2", "LN": 1200},
            {"SN": "chrX", "LN": 800},
            {"SN": "chrY", "LN": 800},
        ],
    }
    rng = np.random.default_rng(7)
    with pysam.AlignmentFile(bam_path, "wb", header=header) as outf:
        read_id = 0
        for tid, length in enumerate([1200, 1200, 800, 800]):
            if tid == 3:
                continue
            for pos in range(10, length - 200, 15):
                tlen = int(rng.integers(150, 180))
                a = pysam.AlignedSegment(outf.header)
                a.query_name = f"r{read_id}"
                a.query_sequence = "A" * 50
                a.flag = 99
                a.reference_id = tid
                a.reference_start = pos
                a.mapping_quality = 60
                a.cigar = ((0, 50),)
                a.next_reference_id = tid
                a.next_reference_start = pos + tlen - 50
                a.template_length = tlen
                a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                outf.write(a)
                read_id += 1
    pysam.index(str(bam_path))
    return bam_path


def test_streaming_cohort_ram_and_memmap_paths(tmp_path: Path) -> None:
    paths = _make_paired_cohort_files(tmp_path, n_samples=8, bin_size=100)
    entries = [
        SampleEntry(filepath=p, sex="M" if i % 2 == 1 else "F")
        for i, p in enumerate(paths)
    ]

    cohort_ram = load_and_assemble_cohort(
        entries,
        target_bin_size=200,
        threads=2,
        max_batch_memory_mb=1,
    )
    assert cohort_ram.counts.shape == (10, 8)
    assert cohort_ram.counts.dtype == np.float32
    assert cohort_ram.counts.flags["F_CONTIGUOUS"]
    assert not isinstance(cohort_ram.counts, np.memmap)
    assert cohort_ram.ref_frag_len is not None
    assert cohort_ram.ref_frag_len.dtype == np.float32
    assert cohort_ram.ref_ep_rate is not None
    assert cohort_ram.ref_ep_rate.dtype == np.float32

    cohort_mmap = load_and_assemble_cohort(
        entries,
        target_bin_size=200,
        threads=4,
        max_batch_memory_mb=0,
    )
    assert isinstance(cohort_mmap.counts, np.memmap)
    assert cohort_mmap.counts.shape == (10, 8)
    assert cohort_mmap.counts.dtype == np.float32
    assert cohort_mmap.counts.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(cohort_mmap.counts, cohort_ram.counts)
    assert cohort_mmap.ref_frag_len is not None
    np.testing.assert_allclose(cohort_mmap.ref_frag_len, cohort_ram.ref_frag_len)
    assert cohort_mmap.ref_ep_rate is not None
    np.testing.assert_allclose(cohort_mmap.ref_ep_rate, cohort_ram.ref_ep_rate)


def test_streaming_cohort_rejects_upsampling_mixed_layout_and_corrupt(
    tmp_path: Path,
) -> None:
    paths = _make_paired_cohort_files(tmp_path, n_samples=2, bin_size=200)
    entries = [SampleEntry(filepath=p) for p in paths]

    with pytest.raises(BinCompatibilityError):
        load_and_assemble_cohort(entries, target_bin_size=100)

    se_path = tmp_path / "single_end.ccnv"
    write_synthetic_sample(
        se_path,
        [20] * 20,
        contigs=["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4,
        starts=[i * 200 for i in range(6)] * 2 + [i * 200 for i in range(4)] * 2,
        ends=[(i + 1) * 200 for i in range(6)] * 2
        + [(i + 1) * 200 for i in range(4)] * 2,
        contig_lengths={"chr1": 1200, "chr2": 1200, "chrX": 800, "chrY": 800},
        bin_size=200,
        fragment_lengths=None,
    )
    with pytest.raises(ReferenceError, match="mixed"):
        load_and_assemble_cohort(
            [SampleEntry(filepath=paths[0]), SampleEntry(filepath=se_path)],
            target_bin_size=200,
        )

    corrupt = tmp_path / "corrupt.ccnv"
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises(ReferenceError):
        load_and_assemble_cohort([SampleEntry(filepath=corrupt)], target_bin_size=200)


def test_prototype_graph_fitting_edge_cases_and_loco_baseline() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((500, 4)).astype(np.float32)
    centers, pids, pweights, mdist = fit_prototype_graph(
        feats, max_prototypes=2048, top_r=4
    )
    assert centers.shape == (8, 4) and centers.dtype == np.float32
    assert pids.shape == (500, 4) and pids.dtype == np.uint16
    assert pweights.shape == (500, 4) and pweights.dtype == np.float16
    assert mdist.shape == (500,) and mdist.dtype == np.float32
    np.testing.assert_allclose(pweights.astype(np.float32).sum(axis=1), 1.0, atol=5e-3)

    # Small U < 8 and U < top_r = 4 edge cases
    c_small, id_small, w_small, d_small = fit_prototype_graph(
        feats[:2], max_prototypes=2048, top_r=4
    )
    assert c_small.shape == (2, 4)
    assert id_small.shape == (2, 4) and id_small.dtype == np.uint16
    assert w_small.shape == (2, 4) and w_small.dtype == np.float16
    assert d_small.shape == (2,)
    np.testing.assert_array_equal(id_small[:, 2:], 0)
    np.testing.assert_array_equal(w_small[:, 2:], 0.0)
    np.testing.assert_allclose(
        w_small[:, :2].astype(np.float32).sum(axis=1), 1.0, atol=5e-3
    )

    c_empty, id_empty, w_empty, d_empty = fit_prototype_graph(
        np.empty((0, 3), dtype=np.float32), max_prototypes=2048, top_r=4
    )
    assert c_empty.shape == (0, 3)
    assert id_empty.shape == (0, 4) and id_empty.dtype == np.uint16
    assert w_empty.shape == (0, 4) and w_empty.dtype == np.float16
    assert d_empty.shape == (0,)

    # LOCO evaluation excludes same-contig gain and never self-contaminates single-contig clusters
    loco_pids = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
        dtype=np.uint16,
    )
    loco_w = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25, 0.0],
        ],
        dtype=np.float16,
    )
    contigs = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    vals_gain0 = np.array([2.0, 2.0, 1.0, 1.0, 1.5], dtype=np.float64)
    baselines_g0, support_g0 = evaluate_loco_prototype_baseline(
        vals_gain0, contigs, loco_pids, loco_w
    )
    np.testing.assert_allclose(baselines_g0[:2], 1.0)
    assert np.all(support_g0[:2] == 2)

    # Primary prototype 1 is 100% on trisomic contig 2 (1.5); secondary prototype 0 is on diploid 0..1 (1.0)
    vals_tri2 = np.array([1.0, 1.0, 1.0, 1.0, 1.5], dtype=np.float64)
    baselines_tri2, support_tri2 = evaluate_loco_prototype_baseline(
        vals_tri2, contigs, loco_pids, loco_w
    )
    assert baselines_tri2[4] == pytest.approx(1.0)
    assert support_tri2[4] == 4

    # Singleton cluster (dists == 0.0) where float16 rounds secondary weights to 0.0 still resolves to 1.0
    singleton_feats = np.vstack(
        [
            rng.normal(0.0, 0.05, size=(12, 2)),
            np.array([[200.0, 200.0]]),
        ]
    ).astype(np.float32)
    _, s_pids, s_weights, _ = fit_prototype_graph(
        singleton_feats, max_prototypes=8, top_r=4
    )
    assert s_weights[12, 0] == np.float16(1.0)
    np.testing.assert_array_equal(s_weights[12, 1:], np.float16(0.0))
    s_contigs = np.array([0] * 6 + [1] * 6 + [2], dtype=np.int32)
    s_vals = np.array([1.0] * 12 + [1.5], dtype=np.float64)
    s_base, s_sup = evaluate_loco_prototype_baseline(
        s_vals, s_contigs, s_pids, s_weights
    )
    assert s_base[12] == pytest.approx(1.0)
    assert s_sup[12] > 0


def test_branch_partitioned_reference_io_and_zero_copy_branch_model(
    tmp_path: Path,
) -> None:
    schema = build_reference_schema(2)
    starts = np.array([0, 100, 0, 100, 0, 100, 0, 0], dtype=np.uint32)
    ends = starts + np.uint32(100)
    n_rows = 8

    meta = ReferenceMetadata(
        bin_size=100,
        contig_lengths={"chr1": 200, "chr2": 200, "chrX": 200, "chrY": 100},
        branches=(BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX),
        autosomal_contigs=("chr1", "chr2"),
        sex_contigs=("chrX", "chrY"),
        pca_components=2,
        branch_pca_ranks={
            BRANCH_AUTOSOMAL: 2,
            BRANCH_FEMALE_SEX: 1,
            BRANCH_MALE_SEX: 0,
        },
    )
    loadings = np.array(
        [
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.7, 0.0],
            [-0.7, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pids = np.zeros((n_rows, 4), dtype=np.uint16)
    pweights = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16), (n_rows, 1))
    branch_dict = pa.DictionaryArray.from_arrays(
        pa.array([0, 0, 0, 0, 1, 1, 2, 2], type=pa.int8()),
        pa.array(
            [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX], type=pa.string()
        ),
    )
    contig_dict = pa.DictionaryArray.from_arrays(
        pa.array([0, 0, 1, 1, 2, 2, 2, 3], type=pa.int16()),
        pa.array(["chr1", "chr2", "chrX", "chrY"], type=pa.string()),
    )
    table = pa.table(
        {
            "branch": branch_dict,
            "contig": contig_dict,
            "start": starts,
            "end": ends,
            "masked": np.zeros(n_rows, dtype=np.bool_),
            "gc": np.full(n_rows, 0.42, dtype=np.float32),
            "mappability": np.ones(n_rows, dtype=np.float32),
            "gc_quantiles": pa.nulls(n_rows, type=pa.list_(pa.float32(), 8)),
            "ref_frag_len": np.full(n_rows, 167.0, dtype=np.float32),
            "ref_ep_rate": np.full(n_rows, 0.1, dtype=np.float32),
            "pca_mean": np.full(n_rows, 0.25, dtype=np.float32),
            "pca_loadings": pa.FixedSizeListArray.from_arrays(
                pa.array(loadings.ravel(), type=pa.float32()), 2
            ),
            "proto_ids": pa.FixedSizeListArray.from_arrays(
                pa.array(pids.ravel(), type=pa.uint16()), 4
            ),
            "proto_weights": pa.FixedSizeListArray.from_arrays(
                pa.array(pweights.ravel(), type=pa.float16()), 4
            ),
            "mean_distance": np.full(n_rows, 0.1, dtype=np.float32),
            "null_mean": np.zeros(n_rows, dtype=np.float32),
            "bio_dispersion": np.full(n_rows, 0.01, dtype=np.float32),
        },
        schema=schema.with_metadata(meta.build_arrow_metadata()),
    )

    ref_path = write_reference_parquet(table, tmp_path / "ref.ccnv")
    pf = pq.ParquetFile(ref_path)
    assert pf.num_row_groups == 3

    auto_only, loaded_meta = read_reference_parquet(
        ref_path, branches=[BRANCH_AUTOSOMAL], check_rows=False
    )
    assert auto_only.num_rows == 4
    assert set(auto_only.column("branch").unique().to_pylist()) == {BRANCH_AUTOSOMAL}
    assert loaded_meta.branch_pca_ranks == {
        BRANCH_AUTOSOMAL: 2,
        BRANCH_FEMALE_SEX: 1,
        BRANCH_MALE_SEX: 0,
    }

    model = build_branch_model(auto_only, BRANCH_AUTOSOMAL, loaded_meta)
    assert model.proto_ids.shape == (4, 4) and model.proto_ids.dtype == np.uint16
    assert (
        model.proto_weights.shape == (4, 4) and model.proto_weights.dtype == np.float16
    )
    assert model.pca_loadings.dtype == np.float32
    assert model.pca_loadings.shape == (2, 4)


def test_end_to_end_reference_and_parallel_bam_predict(tmp_path: Path) -> None:
    cohort_dir = tmp_path / "bam_cohort"
    rng = np.random.default_rng(11)
    contigs = ["chr1"] * 6 + ["chr2"] * 6 + ["chrX"] * 4 + ["chrY"] * 4
    starts = [0, 200, 400, 600, 800, 1000] * 2 + [0, 200, 400, 600] * 2
    ends = [s + 200 for s in starts]
    clens = {"chr1": 1200, "chr2": 1200, "chrX": 800, "chrY": 800}
    sample_paths = []
    for idx in range(6):
        is_male = idx % 2 == 1
        counts = np.array(
            [30 + idx] * 12
            + ([15] * 4 if is_male else [30] * 4)
            + ([12] * 4 if is_male else [0] * 4),
            dtype=np.uint32,
        )
        flens = [
            rng.integers(150, 180, size=int(max(c, 1))).tolist() if c > 0 else []
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

    ref_prefix = tmp_path / "ref"
    ref_path = build_reference(
        ReferenceConfig(
            samples=tuple(sample_paths),
            prefix=ref_prefix,
            bin_size=200,
            threads=2,
            min_cohort_size=4,
            min_sex_group_size=2,
        )
    )
    pf = pq.ParquetFile(ref_path)
    assert pf.num_row_groups == 3

    _, ref_meta = read_reference_parquet(ref_path, branches=[BRANCH_AUTOSOMAL])
    bam_path = _write_synthetic_bam(tmp_path / "query.bam")
    extracted = extract_for_predict(bam_path, ref_meta, threads=2, collect_bands=False)
    assert extracted["frag_bands"] is None
    assert isinstance(extracted["counts"], np.ndarray)
    assert extracted["counts"].shape == (20,)
    assert int(np.sum(extracted["counts"])) > 0

    from unittest.mock import patch

    from ccnv.models import SampleBuffers
    from ccnv.predict import load_inputs

    with patch(
        "ccnv.extract.extracted_to_sample_table",
        side_effect=AssertionError("must not materialize SAMPLE_SCHEMA table"),
    ):
        direct_inputs = load_inputs(bam_path, ref_path, mode="germline", threads=2)
        assert isinstance(direct_inputs.sample, SampleBuffers)
        assert direct_inputs.sample.frag_bands is None

        out_prefix = tmp_path / "pred" / "sample_out"
        res = CliRunner().invoke(
            app,
            [
                "predict",
                str(bam_path),
                str(ref_path),
                "--prefix",
                str(out_prefix),
                "-t",
                "2",
                "--no-bins-vcf",
                "--no-plot",
            ],
        )
    assert res.exit_code == 0, res.output
    assert Path(f"{out_prefix}_bins.bed.gz").is_file()
    assert Path(f"{out_prefix}_segments.bed.gz").is_file()
    assert Path(f"{out_prefix}_qc.json").is_file()
    assert not list((tmp_path / "pred").glob("*.ccnv"))
