"""Unit tests for Schema v2.0.0 Arrow contracts, dictionary/list encoding, and table validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pysam
from helpers import (
    make_prenatal_metadata,
    make_reference_table,
    make_sample_table,
    metadata_for,
    write_test_bam,
)

import ccnv.models as M
from ccnv.extract import extract_sample_table
from ccnv.features.rebinning import rebin_sample_buffers
from ccnv.io.parquet import read_reference_parquet, write_reference_parquet


def _rebin_table(table: pa.Table, target_bin_size: int) -> pa.Table:
    bufs = M.SampleBuffers.from_table(table)
    return rebin_sample_buffers(bufs, target_bin_size).to_table(
        base_metadata=table.schema.metadata
    )


def _sample_meta(
    contig_lengths: dict[str, int],
    bin_size: int,
    *,
    source_identity: dict[str, object] | None = None,
    reference_identity: dict[str, object] | None = None,
    qc: dict[str, object] | None = None,
    contig_order: list[str] | None = None,
) -> dict[bytes, bytes]:
    vals: dict[str, object] = {
        "bin_size": bin_size,
        "source_identity": source_identity or {},
        "reference_identity": reference_identity or {},
        "qc": qc or {},
        "filters": asdict(M.ReadFilters()),
    }
    if contig_order is not None:
        vals["contig_order"] = contig_order
    return metadata_for(
        M.ArtifactKind.SAMPLE, contig_lengths=contig_lengths, values=vals
    )


def _dict_arr(
    indices: Sequence[int] | np.ndarray,
    vocab: Sequence[str | None],
    idx_type: pa.DataType,
    *,
    ordered: bool = False,
) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(indices, idx_type), vocab, ordered=ordered
    )


def _make_valid_reference_table(n_auto: int, n_fem: int, pca_rank: int = 3) -> pa.Table:
    bs = 5_000
    n_tot = n_auto + n_fem
    b_idx = np.concatenate((np.zeros(n_auto, np.int8), np.ones(n_fem, np.int8)))
    c_idx = np.concatenate((np.zeros(n_auto, np.int16), np.ones(n_fem, np.int16)))
    starts = np.concatenate(
        (np.arange(n_auto, dtype=np.uint32), np.arange(n_fem, dtype=np.uint32))
    ) * np.uint32(bs)
    c_lens = {"chr1": n_auto * bs, **({"chrX": n_fem * bs} if n_fem > 0 else {})}
    b_names = [M.BRANCH_AUTOSOMAL, *([M.BRANCH_FEMALE_SEX] if n_fem > 0 else [])]
    meta = metadata_for(
        M.ArtifactKind.REFERENCE,
        contig_lengths=c_lens,
        values={
            "sample_identities": ["s1", "s2"],
            "model": {"bin_size": bs, "branches": b_names, "pca_components": pca_rank},
        },
    )
    ld_col: pa.Array = (
        pa.nulls(n_tot, pa.list_(pa.float32(), 0))
        if pca_rank == 0
        else pa.FixedSizeListArray.from_arrays(
            np.zeros(n_tot * pca_rank, np.float32), pca_rank
        )
    )
    k_p = M.NUM_PROTOTYPES_PER_BIN
    f32 = np.float32
    return pa.table(
        {
            "branch": _dict_arr(
                b_idx, [M.BRANCH_AUTOSOMAL, M.BRANCH_FEMALE_SEX], pa.int8()
            ),
            "contig": _dict_arr(c_idx, ["chr1", "chrX"], pa.int16()),
            "start": starts,
            "end": starts + np.uint32(bs),
            "masked": np.zeros(n_tot, np.bool_),
            "gc": np.full(n_tot, 0.42, f32),
            "mappability": np.ones(n_tot, f32),
            "gc_quantiles": pa.nulls(n_tot, pa.list_(pa.float32(), M.NUM_GC_QUANTILES)),
            "ref_frag_len": np.full(n_tot, 166.0, f32),
            "ref_ep_rate": np.full(n_tot, 2.0, f32),
            "pca_mean": np.ones(n_tot, f32),
            "pca_loadings": ld_col,
            "proto_ids": pa.FixedSizeListArray.from_arrays(
                np.zeros(n_tot * k_p, np.uint16), k_p
            ),
            "proto_weights": pa.FixedSizeListArray.from_arrays(
                np.full(n_tot * k_p, 0.25, np.float16), k_p
            ),
            "mean_distance": np.full(n_tot, 0.1, f32),
            "null_mean": np.zeros(n_tot, f32),
            "bio_dispersion": np.full(n_tot, 0.005, f32),
        },
        schema=M.build_reference_schema(pca_rank).with_metadata(meta),
    )


def _make_read_pair(
    ref_id: int, qname: str, pos: int, tlen: int
) -> tuple[pysam.AlignedSegment, pysam.AlignedSegment]:
    mate_pos = pos + tlen - 50
    seq = "ACGTGCATGC" * 5
    quals = pysam.qualitystring_to_array("I" * 50)
    pair: list[pysam.AlignedSegment] = []
    for flag, start, mstart, tl in (
        (0x1 | 0x2 | 0x20 | 0x40, pos, mate_pos, tlen),
        (0x1 | 0x2 | 0x10 | 0x80, mate_pos, pos, -tlen),
    ):
        rec = pysam.AlignedSegment()
        rec.query_name = qname
        rec.query_sequence = seq
        rec.flag = flag
        rec.reference_id = ref_id
        rec.reference_start = start
        rec.mapping_quality = 60
        rec.cigar = ((0, 50),)
        rec.next_reference_id = ref_id
        rec.next_reference_start = mstart
        rec.template_length = tl
        rec.query_qualities = quals
        pair.append(rec)
    return pair[0], pair[1]


class TestSchemaContracts(unittest.TestCase):
    def test_schema_version_is_2_0_0(self) -> None:
        self.assertEqual(M.SCHEMA_VERSION, "2.0.0")
        self.assertEqual(M.SUPPORTED_SCHEMA_VERSIONS, frozenset({"2.0.0"}))
        self.assertEqual(M.PRENATAL_METADATA_KEY, b"ccnv.prenatal")
        self.assertEqual(M.FRAGMENTOMICS_METADATA_KEY, b"ccnv.fragmentomics")

    def test_sample_schema_has_compact_v2_columns(self) -> None:
        expected = [
            ("count", pa.uint32(), False),
            ("paired_count", pa.uint32(), False),
            ("frag_len_sum", pa.uint64(), False),
            ("short_frag_count", pa.uint32(), False),
            ("endpoint_count", pa.uint32(), False),
            ("frag_gc_sum", pa.float32(), False),
            ("frag_bands", pa.list_(pa.uint32(), M.NUM_FRAG_BANDS), True),
        ]
        self.assertEqual(M.SAMPLE_SCHEMA.names, [f[0] for f in expected])
        for name, exp_type, exp_null in expected:
            fld = M.SAMPLE_SCHEMA.field(name)
            self.assertEqual((fld.type, fld.nullable), (exp_type, exp_null), name)

    def test_sample_contract_round_trip(self) -> None:
        meta = _sample_meta(
            {"chr1": 250_001, "scaffold-A": 99},
            100_000,
            source_identity={"path": "sample.bam"},
            reference_identity={"name": "GRCh-test"},
            qc={"mapped_reads": 42},
        )
        schema = M.SAMPLE_SCHEMA.with_metadata(meta)
        M.validate_schema(schema, M.ArtifactKind.SAMPLE)
        decoded = M.decode_metadata(schema.metadata)
        self.assertIsInstance(
            M.validate_metadata(decoded, M.ArtifactKind.SAMPLE), M.SampleMetadata
        )
        self.assertEqual(decoded["schema_version"], M.SCHEMA_VERSION)
        self.assertEqual(decoded["coordinate_system"], M.COORDINATE_SYSTEM)
        self.assertEqual(decoded["contig_lengths"], {"chr1": 250_001, "scaffold-A": 99})
        self.assertEqual(decoded["contig_order"], ["chr1", "scaffold-A"])

    def test_schema_rejects_invalid_coordinate_system(self) -> None:
        values = M.decode_metadata(_sample_meta({"chr1": 100}, 100))
        values["coordinate_system"] = "1-based-closed"
        invalid = {
            f"ccnv.{k}".encode(): json.dumps(v).encode() for k, v in values.items()
        }
        with self.assertRaisesRegex(M.SchemaError, "0-based-half-open"):
            M.validate_schema(
                M.SAMPLE_SCHEMA.with_metadata(invalid), M.ArtifactKind.SAMPLE
            )

    def test_schema_rejects_wrong_arrow_fields(self) -> None:
        schema = pa.schema(
            [
                pa.field("contig", pa.string(), False),
                pa.field("start", pa.int64(), False),
            ]
        )
        with self.assertRaisesRegex(M.SchemaError, "fields do not match"):
            M.validate_schema(schema, M.ArtifactKind.SAMPLE)

    def test_table_accepts_partial_final_bin_and_arbitrary_contigs(self) -> None:
        table = make_sample_table(
            [10, 11, 12],
            contigs=["scaffold-A"] * 3,
            starts=[0, 100_000, 200_000],
            ends=[100_000, 200_000, 250_001],
            contig_lengths={"scaffold-A": 250_001},
            bin_size=100_000,
        )
        codes, names, starts, ends = M.reconstruct_sample_grid(
            M.validate_sample_table(table)
        )
        self.assertEqual(names, ("scaffold-A",))
        self.assertEqual(codes.tolist(), [0, 0, 0])
        self.assertEqual(starts.tolist(), [0, 100_000, 200_000])
        self.assertEqual(ends.tolist(), [100_000, 200_000, 250_001])

    def test_table_rejects_mismatched_sample_row_count(self) -> None:
        table = make_sample_table([1, 2], contig_lengths={"chr1": 10}, bin_size=10)
        with self.assertRaisesRegex(M.SchemaError, "outside"):
            M.validate_sample_table(table)

    def test_bin_coordinates_are_contig_start_end_tuples(self) -> None:
        table = make_sample_table(
            [1, 2],
            contigs=["chr1", "chr2"],
            starts=[0, 0],
            ends=[100, 80],
            contig_lengths={"chr1": 100, "chr2": 80},
            bin_size=100,
        )
        codes, names, starts, ends = M.reconstruct_sample_grid(
            M.validate_sample_table(table)
        )
        coords = tuple(
            (names[int(c)], int(s), int(e))
            for c, s, e in zip(codes, starts, ends, strict=True)
        )
        self.assertEqual(coords, (("chr1", 0, 100), ("chr2", 0, 80)))

    def test_reference_schema_accepted_and_old_rejected(self) -> None:
        ref_table = make_reference_table(
            branches=["autosomal", "autosomal"],
            contigs=["chr1", "chr1"],
            starts=[0, 100],
            ends=[100, 200],
            contig_lengths={"chr1": 200},
            pca_loadings=[[0.1, 0.1], [0.1, 0.1]],
            model={"bin_size": 100, "branches": ["autosomal"]},
        )
        ref_meta = ref_table.schema.metadata or {}
        M.validate_schema(ref_table.schema, M.ArtifactKind.REFERENCE)
        self.assertIsInstance(
            M.validate_metadata(M.decode_metadata(ref_meta), M.ArtifactKind.REFERENCE),
            M.ReferenceMetadata,
        )
        self.assertIsInstance(
            M.validate_reference_table(ref_table), M.ReferenceMetadata
        )

        old_fields = [
            ("branch", pa.string()),
            ("contig", pa.string()),
            ("start", pa.int64()),
            ("end", pa.int64()),
            ("masked", pa.bool_()),
            ("pca_mean", pa.float64()),
            ("pca_loadings", pa.list_(pa.float64())),
            ("nearest_ref_indices", pa.list_(pa.int32())),
        ]
        old_schema = pa.schema([pa.field(n, t, False) for n, t in old_fields])
        with self.assertRaisesRegex(M.SchemaError, "fields do not match"):
            M.validate_schema(
                old_schema.with_metadata(ref_meta), M.ArtifactKind.REFERENCE
            )

    def test_validate_reference_table_all_five_branches(self) -> None:
        prenatal_meta = make_prenatal_metadata()
        branches_list = [
            M.BRANCH_AUTOSOMAL,
            M.BRANCH_AUTOSOMAL,
            M.BRANCH_FEMALE_SEX,
            M.BRANCH_MALE_SEX,
            M.BRANCH_PRENATAL_AUTOSOMAL,
            M.BRANCH_PRENATAL_AUTOSOMAL,
            M.BRANCH_PRENATAL_FFY,
        ]
        table = make_reference_table(
            branches=branches_list,
            contigs=["chr1", "chr1", "chrX", "chrY", "chr1", "chr1", "chrY"],
            starts=[0, 5000, 0, 0, 0, 100_000, 0],
            ends=[5000, 10_000, 5000, 5000, 100_000, 200_000, 5000],
            contig_lengths={"chr1": 200_000, "chrX": 100_000, "chrY": 100_000},
            pca_loadings=[[0.1]] * len(branches_list),
            sample_identities=["s1", "s2"],
            model={
                "bin_size": 5000,
                "prenatal_bin_size": 100_000,
                "mode": "prenatal",
                "branches": sorted(M.KNOWN_REFERENCE_BRANCHES),
            },
            extra_metadata={"prenatal": prenatal_meta.to_dict()},
        )
        M.validate_reference_table(table)

        extracted_prenatal = M.PrenatalModelMetadata.from_table(table)
        if extracted_prenatal is None:
            self.fail("expected non-None extracted_prenatal")
        self.assertEqual(
            extracted_prenatal.yfrac_threshold, prenatal_meta.yfrac_threshold
        )

        ref_obj = M.ReferenceMetadata.from_table(table)
        self.assertEqual(
            (ref_obj.mode, ref_obj.bin_size, ref_obj.prenatal_bin_size),
            ("prenatal", 5000, 100_000),
        )

    def test_validate_reference_table_rejects_unknown_branch(self) -> None:
        table = make_reference_table(
            branches=["unknown_branch"],
            contigs=["chr1"],
            starts=[0],
            ends=[5000],
            contig_lengths={"chr1": 10_000},
            pca_loadings=[[0.1]],
            model={"bin_size": 5000, "branches": ["unknown_branch"]},
        )
        with self.assertRaisesRegex(M.SchemaError, "unknown branch"):
            M.validate_reference_table(table)

    def test_validate_reference_table_bin_spacing_checks(self) -> None:
        both = {
            "bin_size": 5000,
            "prenatal_bin_size": 100_000,
            "branches": [M.BRANCH_AUTOSOMAL, M.BRANCH_PRENATAL_AUTOSOMAL],
        }
        no_psize = {"bin_size": 5000, "branches": [M.BRANCH_PRENATAL_AUTOSOMAL]}
        cases: list[tuple[str, int, int, dict[str, object], str]] = [
            (M.BRANCH_AUTOSOMAL, 0, 10_000, both, "bin width 10000 does not match"),
            (
                M.BRANCH_PRENATAL_AUTOSOMAL,
                0,
                5000,
                both,
                "bin width 5000 does not match",
            ),
            (
                M.BRANCH_PRENATAL_AUTOSOMAL,
                0,
                100_000,
                no_psize,
                "prenatal_bin_size is not specified",
            ),
            (
                M.BRANCH_AUTOSOMAL,
                100,
                5100,
                both,
                "start coordinate 100 is not a multiple",
            ),
        ]
        for branch, start, end, model, pattern in cases:
            tbl = make_reference_table(
                branches=[branch],
                contigs=["chr1"],
                starts=[start],
                ends=[end],
                contig_lengths={"chr1": 200_000},
                pca_loadings=[[0.1]],
                model=model,
            )
            with (
                self.subTest(pattern=pattern),
                self.assertRaisesRegex(M.SchemaError, pattern),
            ):
                M.validate_reference_table(tbl)

    def test_validate_reference_table_large_table_fast_path(self) -> None:
        n_bins = 1005
        starts = [i * 5000 for i in range(n_bins)]
        ends = [(i + 1) * 5000 for i in range(n_bins)]

        def _tbl(contigs: list[str]) -> pa.Table:
            return make_reference_table(
                branches=[M.BRANCH_AUTOSOMAL] * n_bins,
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths={"chr1": n_bins * 5000},
                pca_loadings=[[0.1]] * n_bins,
                sample_identities=["s1"],
                model={"bin_size": 5000, "branches": [M.BRANCH_AUTOSOMAL]},
            )

        M.validate_reference_table(_tbl(["chr1"] * n_bins))
        with self.assertRaisesRegex(M.SchemaError, "undeclared contig 'chrUndeclared'"):
            M.validate_reference_table(
                _tbl(["chr1"] * (n_bins - 1) + ["chrUndeclared"])
            )

    def test_legacy_v1_table_rejected_with_version_error(self) -> None:
        base = {"schema_version": "1.0.0", "coordinate_system": M.COORDINATE_SYSTEM}
        s_meta = M.encode_metadata(
            {
                "artifact_kind": "sample",
                **base,
                "contig_lengths": {"chr1": 10_000},
                "bin_size": 5_000,
                "source_identity": {"path": "s.bam", "format": "bam"},
                "reference_identity": {"n_contigs": 1, "contig_order": ["chr1"]},
                "qc": {},
                "filters": asdict(M.ReadFilters()),
            }
        )
        v1_tbl = pa.table(
            {
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 5_000], pa.int64()),
                "end": pa.array([5_000, 10_000], pa.int64()),
                "count": pa.array([10, 12], pa.uint64()),
                "fragment_lengths": pa.array([[166], [170]], pa.list_(pa.uint16())),
                "endpoint_count": pa.array([20, 24], pa.uint32()),
                "frag_gc_sum": pa.array([4.2, 5.1], pa.float32()),
            },
            metadata=s_meta,
        )
        with self.assertRaises(M.SchemaVersionError):
            M.validate_sample_table(v1_tbl)

        r_meta = M.encode_metadata(
            {
                "artifact_kind": "reference",
                **base,
                "contig_lengths": {"chr1": 5_000},
                "sample_identities": ["s1"],
                "model": {"bin_size": 5_000, "branches": ["autosomal"]},
            }
        )
        with self.assertRaises(M.SchemaVersionError):
            M.validate_reference_table(
                pa.table({"branch": ["autosomal"]}, metadata=r_meta)
            )

    def test_reference_schema_v2_purged_prenatal_and_fast_vectorized_validation(
        self,
    ) -> None:
        prenatal_fields = set(M.PrenatalModelMetadata.__dataclass_fields__.keys())
        for dead_prefix in ("mlp_", "olm_skew_", "preface_"):
            self.assertFalse(any(f.startswith(dead_prefix) for f in prenatal_fields))
        removed_names = (
            "preface_std",
            "preface_logit_mode",
            "ffy_intercept",
            "ffx_intercept",
            "ffx_slope",
            "ffx_std",
            "alpha_0",
            "sigma_x_sys",
        )
        for removed in removed_names:
            self.assertNotIn(removed, prenatal_fields)

        ref_r0 = _make_valid_reference_table(20, 10, pca_rank=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_pq = write_reference_parquet(ref_r0, Path(tmpdir) / "ref_r0.ccnv")
            reloaded_r0, meta_r0 = read_reference_parquet(ref_pq, check_rows=True)
            self.assertEqual((meta_r0.bin_size, reloaded_r0.num_rows), (5_000, 30))

        ref_large = _make_valid_reference_table(50_000, 10_000, pca_rank=4)
        M.validate_reference_table(ref_large)
        t0 = perf_counter()
        ref_meta = M.validate_reference_table(ref_large, check_rows=True)
        elapsed_ms = (perf_counter() - t0) * 1_000.0
        self.assertEqual(ref_meta.branches, (M.BRANCH_AUTOSOMAL, M.BRANCH_FEMALE_SEX))
        self.assertLess(elapsed_ms, 25.0, f"validation took {elapsed_ms:.2f} ms")

        bad_starts = ref_r0["start"].to_numpy().copy()
        bad_starts[5] += 5_000
        bad_gap = ref_r0.set_column(2, "start", pa.array(bad_starts, type=pa.uint32()))
        with self.assertRaises(M.SchemaError):
            M.validate_reference_table(bad_gap, check_rows=True)
        M.validate_reference_table(bad_gap, check_rows=False)

        non_contig_b = np.array([0] * 10 + [1] * 10 + [0] * 10, dtype=np.int8)
        bad_runs = ref_r0.set_column(
            0,
            "branch",
            pa.DictionaryArray.from_arrays(
                non_contig_b, [M.BRANCH_AUTOSOMAL, M.BRANCH_FEMALE_SEX]
            ),
        )
        with self.assertRaises(M.SchemaError):
            M.validate_reference_table(bad_runs, check_rows=True)

    def test_dictionary_array_duplicate_vocab_and_null_dictionary_rejected(
        self,
    ) -> None:
        base_ref = _make_valid_reference_table(2, 0, pca_rank=1)
        dup_c = _dict_arr([0, 1], ["chr1", "chr1"], pa.int16())
        dup_c_tbl = (
            base_ref.set_column(1, "contig", dup_c)
            .set_column(2, "start", pa.array([0, 0], pa.uint32()))
            .set_column(3, "end", pa.array([5_000, 5_000], pa.uint32()))
        )
        with self.assertRaises(M.SchemaError):
            M.validate_reference_table(dup_c_tbl, check_rows=True)

        ref_3 = _make_valid_reference_table(2, 1, pca_rank=1)
        dup_b = _dict_arr(
            [0, 1, 2],
            [M.BRANCH_AUTOSOMAL, M.BRANCH_FEMALE_SEX, M.BRANCH_AUTOSOMAL],
            pa.int8(),
        )
        dup_b_tbl = (
            ref_3.set_column(0, "branch", dup_b)
            .set_column(1, "contig", _dict_arr([0, 1, 0], ["chr1", "chrX"], pa.int16()))
            .set_column(2, "start", pa.array([0, 0, 5_000], pa.uint32()))
            .set_column(3, "end", pa.array([5_000, 5_000, 10_000], pa.uint32()))
        )
        with self.assertRaisesRegex(M.SchemaError, "contiguous"):
            M.validate_reference_table(dup_b_tbl, check_rows=True)

        for col_idx, col_name, null_arr in (
            (0, "branch", _dict_arr([0, 0], [M.BRANCH_AUTOSOMAL, None], pa.int8())),
            (1, "contig", _dict_arr([0, 1], ["chr1", None], pa.int16())),
        ):
            with self.assertRaisesRegex(M.SchemaError, "dictionary"):
                M.validate_reference_table(
                    base_ref.set_column(col_idx, col_name, null_arr), check_rows=True
                )

    def test_fixed_size_list_inner_nulls_rejected_on_sample_and_reference(self) -> None:
        b_type = pa.list_(pa.uint32(), M.NUM_FRAG_BANDS)
        bad_bands = pa.array(
            [[1, None, *([0] * (M.NUM_FRAG_BANDS - 2))], [0] * M.NUM_FRAG_BANDS],
            type=b_type,
        )
        self.assertEqual(bad_bands.null_count, 0)
        bad_sample = make_sample_table(
            [5, 5], contig_lengths={"chr1": 10_000}, bin_size=5_000
        ).set_column(6, "frag_bands", bad_bands)
        msg_fb = "column 'frag_bands' contains null list elements"
        with self.assertRaisesRegex(M.SchemaError, msg_fb):
            M.validate_sample_table(bad_sample)
        with self.assertRaisesRegex(M.SchemaError, msg_fb):
            _rebin_table(bad_sample, 10_000)

        mixed_bands = pa.array(
            [None, [1, None, *([0] * (M.NUM_FRAG_BANDS - 2))]], type=b_type
        )
        with self.assertRaisesRegex(M.SchemaError, msg_fb):
            M.validate_sample_table(bad_sample.set_column(6, "frag_bands", mixed_bands))

        clean_null = pa.array([None, [2] * M.NUM_FRAG_BANDS], type=b_type)
        coarsened = _rebin_table(
            bad_sample.set_column(6, "frag_bands", clean_null), 10_000
        )
        self.assertEqual(coarsened["frag_bands"][0].as_py(), [2] * M.NUM_FRAG_BANDS)

        base_ref = _make_valid_reference_table(2, 0, pca_rank=2)
        for col_name, bad_col in (
            (
                "proto_ids",
                pa.array(
                    [[0, None, 1, 2], [0, 1, 2, 3]],
                    pa.list_(pa.uint16(), M.NUM_PROTOTYPES_PER_BIN),
                ),
            ),
            (
                "gc_quantiles",
                pa.array(
                    [None, [0.1, None, *([0.1] * (M.NUM_GC_QUANTILES - 2))]],
                    pa.list_(pa.float32(), M.NUM_GC_QUANTILES),
                ),
            ),
        ):
            idx = base_ref.schema.get_field_index(col_name)
            with self.assertRaisesRegex(
                M.SchemaError, f"column '{col_name}' contains null list elements"
            ):
                M.validate_reference_table(base_ref.set_column(idx, col_name, bad_col))

    def test_fixed_size_list_size_validation_sliced_rebinning_and_multichunk_null_dict(
        self,
    ) -> None:
        t_12 = pa.list_(pa.uint32(), 12)
        arr_12 = pa.FixedSizeListArray.from_arrays(
            pa.array(np.arange(24, dtype=np.uint32)), 12
        )
        for bad in (
            arr_12,
            pa.chunked_array([arr_12]),
            pa.array([], t_12),
            pa.chunked_array([], t_12),
            pa.nulls(3, t_12),
        ):
            with self.assertRaisesRegex(M.SchemaError, r"FixedSizeList\[24\].*\[12\]"):
                M.fixed_size_list_to_numpy(bad, 24, np.uint32)
        with self.assertRaisesRegex(M.SchemaError, "FixedSizeListType"):
            M.fixed_size_list_to_numpy(pa.array([], type=pa.uint32()), 24, np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bam_path = tmp / "slice_paired.bam"
            contigs = [("chr1", 100_000), ("chr2", 80_000)]
            rng = np.random.default_rng(17)
            records: list[pysam.AlignedSegment] = []
            read_idx = 0
            for ref_id, (_, length) in enumerate(contigs):
                for p_raw, t_raw in zip(
                    rng.integers(100, max(200, length - 400), size=24),
                    rng.integers(90, 260, size=24),
                    strict=True,
                ):
                    records.extend(
                        _make_read_pair(
                            ref_id, f"frag_{read_idx}", int(p_raw), int(t_raw)
                        )
                    )
                    read_idx += 1

            write_test_bam(bam_path, records, contigs=contigs, index=True)
            full_5k = extract_sample_table(
                M.ExtractConfig(alignment=bam_path, prefix=tmp / "s", bin_size=5_000)
            )
            chr2_meta = _sample_meta(
                {"chr2": 80_000},
                5_000,
                contig_order=["chr2"],
                source_identity={"path": str(bam_path)},
                reference_identity={"n_contigs": 1, "contig_order": ["chr2"]},
            )
            sliced_chr2 = full_5k.slice(20, 16).replace_schema_metadata(chr2_meta)
            self.assertEqual(sliced_chr2.column("frag_bands").chunk(0).offset, 20)
            coarsened = _rebin_table(sliced_chr2, 20_000)
            self.assertEqual(coarsened.num_rows, 4)
            exp_b = M.fixed_size_list_to_numpy(
                sliced_chr2.column("frag_bands"), M.NUM_FRAG_BANDS, np.uint32
            )
            act_b = M.fixed_size_list_to_numpy(
                coarsened.column("frag_bands"), M.NUM_FRAG_BANDS, np.uint32
            )
            np.testing.assert_array_equal(act_b.sum(axis=0), exp_b.sum(axis=0))

        base_ref = _make_valid_reference_table(2, 0, pca_rank=1)
        b_mc = pa.chunked_array(
            [
                _dict_arr([0], [M.BRANCH_AUTOSOMAL, None], pa.int8()),
                _dict_arr([0], [M.BRANCH_AUTOSOMAL], pa.int8()),
            ]
        )
        with self.assertRaisesRegex(M.SchemaError, "dictionary"):
            M.validate_reference_table(
                base_ref.set_column(0, "branch", b_mc), check_rows=True
            )

        c_mc = pa.chunked_array(
            [
                _dict_arr([0], ["chr1", None], pa.int16()),
                _dict_arr([0], ["chr1"], pa.int16()),
            ]
        )
        mc_null_c = base_ref.set_column(1, "contig", c_mc)
        with self.assertRaisesRegex(M.SchemaError, "dictionary"):
            M.validate_reference_table(mc_null_c, check_rows=True)
        with self.assertRaisesRegex(M.SchemaError, "dictionary"):
            M.decode_dictionary_contigs(mc_null_c.column("contig"))

    def test_canonical_dictionary_array_contracts(self) -> None:
        ordered = _dict_arr(
            [0, 1], ["autosomal", "female_sex"], pa.int8(), ordered=True
        )
        canon_ord = M.canonical_dictionary_array(
            ordered, "branch", index_type=pa.int8()
        )
        self.assertEqual(
            (canon_ord.type, canon_ord.type.ordered),
            (pa.dictionary(pa.int8(), pa.string()), False),
        )

        canon_lg = M.canonical_dictionary_array(
            pa.array(["chr1", "chr2"], pa.large_string()),
            "contig",
            index_type=pa.int16(),
        )
        self.assertEqual(canon_lg.type, pa.dictionary(pa.int16(), pa.string()))

        canon_128 = M.canonical_dictionary_array(
            pa.array([f"b_{i}" for i in range(128)]), "branch", index_type=pa.int8()
        )
        self.assertEqual(len(canon_128.dictionary), 128)
        with self.assertRaises(M.SchemaError):
            M.canonical_dictionary_array(
                pa.array([f"b_{i}" for i in range(129)]), "branch", index_type=pa.int8()
            )

    def test_int16_contig_dictionary_bound_32768(self) -> None:
        def _grid_meta(count: int) -> M.SampleMetadata:
            names = tuple(f"c{i}" for i in range(count))
            return M.SampleMetadata(
                schema_version=M.SCHEMA_VERSION,
                coordinate_system=M.COORDINATE_SYSTEM,
                bin_size=5000,
                contig_order=names,
                contig_lengths={c: 5000 for c in names},
                source_identity={"id": str(count)},
                reference_identity={"name": "ref"},
                filters={},
                qc={},
            )

        codes, decoded_names, starts, ends = M.reconstruct_sample_grid(
            _grid_meta(32_768)
        )
        self.assertEqual(
            (len(decoded_names), int(codes[-1]), starts.size, ends.size),
            (32_768, 32_767, 32_768, 32_768),
        )

        meta_bad = _grid_meta(32_769)
        with self.assertRaises(M.SchemaError):
            M.reconstruct_sample_grid(meta_bad)

        n_bad = len(meta_bad.contig_order)
        z_f32 = np.zeros(n_bad, dtype=np.float32)
        fit_bad = M.BranchFitArrays(
            branch=M.BRANCH_AUTOSOMAL,
            contigs=meta_bad.contig_order,
            starts=np.zeros(n_bad, dtype=np.int64),
            ends=np.full(n_bad, 5000, dtype=np.int64),
            masked=np.zeros(n_bad, dtype=np.bool_),
            gc=np.full(n_bad, 0.42, dtype=np.float32),
            mappability=np.ones(n_bad, dtype=np.float32),
            gc_quantiles=np.zeros((n_bad, M.NUM_GC_QUANTILES), dtype=np.float32),
            ref_frag_len=z_f32,
            ref_ep_rate=z_f32,
            pca_mean=z_f32,
            pca_loadings=np.zeros((n_bad, 0), dtype=np.float32),
            proto_ids=np.zeros((n_bad, M.NUM_PROTOTYPES_PER_BIN), dtype=np.uint16),
            proto_weights=np.zeros((n_bad, M.NUM_PROTOTYPES_PER_BIN), dtype=np.float16),
            mean_distance=z_f32,
            null_mean=np.ones(n_bad, dtype=np.float32),
            bio_dispersion=np.full(n_bad, 1e-3, dtype=np.float32),
        )
        with self.assertRaises(M.SchemaError):
            fit_bad.to_table()


if __name__ == "__main__":
    unittest.main()
