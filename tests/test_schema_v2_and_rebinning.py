"""Tests for Schema v2.0.0, domain/config fixes, vectorized validation, and sketch rebinning."""

from __future__ import annotations

import logging
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pysam

from ccnv.extract import extract_sample_table
from ccnv.features.rebinning import rebin_sample_sketch
from ccnv.io.parquet import (
    read_reference_parquet,
    read_sample_parquet,
    write_reference_parquet,
    write_sample_parquet,
)
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    COORDINATE_SYSTEM,
    FRAGMENTOMICS_METADATA_KEY,
    NUM_FRAG_BANDS,
    NUM_GC_QUANTILES,
    NUM_PROTOTYPES_PER_BIN,
    SAMPLE_SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ArtifactKind,
    BinCompatibilityError,
    ExtractConfig,
    PredictConfig,
    PrenatalModelMetadata,
    ReadFilters,
    ReferenceConfig,
    ReferenceMetadata,
    SampleMetadata,
    SchemaError,
    SchemaVersionError,
    build_reference_schema,
    decode_metadata,
    encode_metadata,
    metadata_for,
    parquet_path,
    reconstruct_sample_grid,
    resolve_contig_name,
    select_layout,
)
from ccnv.models.validation import (
    validate_metadata,
    validate_reference_table,
    validate_sample_table,
    validate_schema,
)


def _write_paired_bam(path: Path, contigs: list[tuple[str, int]]) -> None:
    """Create an indexed paired-end BAM with proper pairs across contigs."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs],
    }
    records: list[pysam.AlignedSegment] = []
    rng = np.random.default_rng(17)
    read_idx = 0
    for ref_id, (_, length) in enumerate(contigs):
        positions = rng.integers(100, max(200, length - 400), size=24)
        inserts = rng.integers(90, 260, size=24)
        for pos_raw, tlen_raw in zip(positions, inserts, strict=True):
            pos = int(pos_raw)
            tlen = int(tlen_raw)
            mate_pos = pos + tlen - 50
            seq = "ACGTGCATGC" * 5

            r1 = pysam.AlignedSegment()
            r1.query_name = f"frag_{read_idx}"
            r1.query_sequence = seq
            r1.flag = 0x1 | 0x2 | 0x20 | 0x40
            r1.reference_id = ref_id
            r1.reference_start = pos
            r1.mapping_quality = 60
            r1.cigar = ((0, 50),)
            r1.next_reference_id = ref_id
            r1.next_reference_start = mate_pos
            r1.template_length = tlen
            r1.query_qualities = pysam.qualitystring_to_array("I" * 50)

            r2 = pysam.AlignedSegment()
            r2.query_name = f"frag_{read_idx}"
            r2.query_sequence = seq
            r2.flag = 0x1 | 0x2 | 0x10 | 0x80
            r2.reference_id = ref_id
            r2.reference_start = mate_pos
            r2.mapping_quality = 60
            r2.cigar = ((0, 50),)
            r2.next_reference_id = ref_id
            r2.next_reference_start = pos
            r2.template_length = -tlen
            r2.query_qualities = pysam.qualitystring_to_array("I" * 50)

            records.extend((r1, r2))
            read_idx += 1

    records.sort(key=lambda rec: (rec.reference_id, rec.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for rec in records:
            bam.write(rec)
    pysam.index(str(path))


def _write_single_end_bam(path: Path, contigs: list[tuple[str, int]]) -> None:
    """Create an indexed single-end BAM."""
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in contigs],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for idx, pos in enumerate((50, 5_100, 12_300)):
            rec = pysam.AlignedSegment()
            rec.query_name = f"se_{idx}"
            rec.query_sequence = "ACGTACGTAC" * 5
            rec.flag = 0
            rec.reference_id = 0
            rec.reference_start = pos
            rec.mapping_quality = 60
            rec.cigar = ((0, 50),)
            rec.next_reference_id = -1
            rec.next_reference_start = -1
            rec.template_length = 0
            rec.query_qualities = pysam.qualitystring_to_array("I" * 50)
            bam.write(rec)
    pysam.index(str(path))


def _make_valid_reference_table(n_auto: int, n_fem: int, pca_rank: int = 3) -> pa.Table:
    """Construct a valid Schema v2.0.0 dictionary-encoded reference table."""
    bin_size = 5_000
    n_total = n_auto + n_fem
    branch_indices = np.concatenate(
        (np.zeros(n_auto, dtype=np.int8), np.ones(n_fem, dtype=np.int8))
    )
    contig_indices = np.concatenate(
        (np.zeros(n_auto, dtype=np.int16), np.ones(n_fem, dtype=np.int16))
    )
    starts = np.concatenate(
        (
            np.arange(n_auto, dtype=np.uint32) * np.uint32(bin_size),
            np.arange(n_fem, dtype=np.uint32) * np.uint32(bin_size),
        )
    )
    ends = starts + np.uint32(bin_size)

    contig_lengths = {"chr1": n_auto * bin_size}
    branches = [BRANCH_AUTOSOMAL]
    if n_fem > 0:
        contig_lengths["chrX"] = n_fem * bin_size
        branches.append(BRANCH_FEMALE_SEX)
    meta = metadata_for(
        ArtifactKind.REFERENCE,
        contig_lengths=contig_lengths,
        values={
            "sample_identities": ["s1", "s2"],
            "model": {
                "bin_size": bin_size,
                "branches": branches,
                "pca_components": pca_rank,
            },
        },
    )
    schema = build_reference_schema(pca_rank).with_metadata(meta)
    loadings_col: pa.Array = (
        pa.nulls(n_total, type=pa.list_(pa.float32(), 0))
        if pca_rank == 0
        else pa.FixedSizeListArray.from_arrays(
            np.zeros(n_total * pca_rank, dtype=np.float32), pca_rank
        )
    )
    return pa.table(
        {
            "branch": pa.DictionaryArray.from_arrays(
                branch_indices, [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX]
            ),
            "contig": pa.DictionaryArray.from_arrays(contig_indices, ["chr1", "chrX"]),
            "start": starts,
            "end": ends,
            "masked": np.zeros(n_total, dtype=np.bool_),
            "gc": np.full(n_total, 0.42, dtype=np.float32),
            "mappability": np.ones(n_total, dtype=np.float32),
            "gc_quantiles": pa.nulls(
                n_total, type=pa.list_(pa.float32(), NUM_GC_QUANTILES)
            ),
            "ref_frag_len": np.full(n_total, 166.0, dtype=np.float32),
            "ref_ep_rate": np.full(n_total, 2.0, dtype=np.float32),
            "pca_mean": np.ones(n_total, dtype=np.float32),
            "pca_loadings": loadings_col,
            "proto_ids": pa.FixedSizeListArray.from_arrays(
                np.zeros(n_total * NUM_PROTOTYPES_PER_BIN, dtype=np.uint16),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "proto_weights": pa.FixedSizeListArray.from_arrays(
                np.full(n_total * NUM_PROTOTYPES_PER_BIN, 0.25, dtype=np.float16),
                NUM_PROTOTYPES_PER_BIN,
            ),
            "mean_distance": np.full(n_total, 0.1, dtype=np.float32),
            "null_mean": np.zeros(n_total, dtype=np.float32),
            "bio_dispersion": np.full(n_total, 0.005, dtype=np.float32),
        },
        schema=schema,
    )


class TestSchemaV2AndRebinning(unittest.TestCase):
    def test_canine_38_autosomes_preserved_when_all_contigs_false(self) -> None:
        header = [(f"chr{i}", 100_000) for i in range(1, 39)] + [
            ("chrX", 100_000),
            ("chrM", 16_000),
            ("chr0", 5_000),
        ]
        selected = select_layout(header, all_contigs=False)
        assert len(selected) == 39
        assert ("chr38", 100_000) in selected
        assert ("chrM", 16_000) not in selected
        assert ("chr0", 5_000) not in selected

        legacy_contigs = select_layout(
            [name for name, _ in header],
            [length for _, length in header],
            all_contigs=False,
        )
        assert len(legacy_contigs) == 39
        assert legacy_contigs[-2].name == "chr38"

    def test_resolve_contig_name_handles_single_pass_generator_and_config_preserves_prefix(
        self,
    ) -> None:
        assert resolve_contig_name("CHR1", (c for c in ["chr1", "chr2"])) == "chr1"
        cfg = ReferenceConfig(samples=Path("sheet.yaml"), prefix=Path("out/ref"))
        assert cfg.prefix == Path("out/ref")

        p_cfg = PredictConfig(
            sample=Path("s.bam"),
            reference=Path("ref.ccnv"),
            prefix=Path("out/pred"),
        )
        assert p_cfg.threads == 1
        assert p_cfg.bins_vcf is False

        with self.assertRaises(ValueError):
            ReferenceConfig(prefix=Path("out/ref"), bin_size=0)
        with self.assertRaises(ValueError):
            ReferenceConfig(prefix=Path("out/ref"), min_mappability=1.2)
        with self.assertRaises(ValueError):
            PredictConfig(
                sample=Path("s.bam"),
                reference=Path("r.ccnv"),
                prefix=Path("o"),
                bin_size=0,
            )
        with self.assertRaises(ValueError):
            PredictConfig(
                sample=Path("s.bam"),
                reference=Path("r.ccnv"),
                prefix=Path("o"),
                min_mappability=-0.1,
            )
        with self.assertRaises(ValueError):
            PredictConfig(
                sample=Path("s.bam"),
                reference=Path("r.ccnv"),
                prefix=Path("o"),
                threads=0,
            )

    def test_sample_sketch_v2_roundtrip_nested_coarsening_and_upsample_rejection(
        self,
    ) -> None:
        assert SCHEMA_VERSION == "2.0.0"
        assert SUPPORTED_SCHEMA_VERSIONS == frozenset({"2.0.0"})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bam_path = tmp / "paired.bam"
            # Non-multiple contig lengths test partial final bins across contigs
            _write_paired_bam(bam_path, [("chr1", 250_000), ("chr2", 105_000)])

            table_5k = extract_sample_table(
                ExtractConfig(
                    alignment=bam_path,
                    prefix=tmp / "s",
                    bin_size=5_000,
                    threads=2,
                )
            )
            assert "contig" not in table_5k.column_names
            assert "start" not in table_5k.column_names
            assert "end" not in table_5k.column_names
            assert table_5k.schema.equals(SAMPLE_SCHEMA, check_metadata=False)
            assert table_5k.schema.field("frag_bands").type == pa.list_(
                pa.uint32(), NUM_FRAG_BANDS
            )
            assert table_5k["frag_bands"].null_count == 0

            # Parquet roundtrip and grid reconstruction
            out_pq = write_sample_parquet(table_5k, parquet_path(tmp / "s"))
            reloaded = read_sample_parquet(out_pq)
            meta_5k = validate_sample_table(reloaded)
            assert isinstance(meta_5k, SampleMetadata)
            assert meta_5k.contig_order == ("chr1", "chr2")
            codes, names, starts, ends = reconstruct_sample_grid(meta_5k)
            assert names == ("chr1", "chr2")
            assert codes.dtype == np.int16
            assert starts.dtype == np.uint32 and ends.dtype == np.uint32
            assert codes.size == table_5k.num_rows == 50 + 21

            # Nested integer-multiple coarsening preserves all 7 channels & fragmentomics
            table_100k = rebin_sample_sketch(table_5k, 100_000)
            meta_100k = validate_sample_table(table_100k)
            assert meta_100k.bin_size == 100_000
            assert table_100k.num_rows == 3 + 2
            assert FRAGMENTOMICS_METADATA_KEY in (table_100k.schema.metadata or {})
            for col_name in (
                "count",
                "paired_count",
                "frag_len_sum",
                "short_frag_count",
                "endpoint_count",
            ):
                assert int(pc.sum(table_100k[col_name]).as_py()) == int(
                    pc.sum(table_5k[col_name]).as_py()
                )
            np.testing.assert_allclose(
                float(pc.sum(table_100k["frag_gc_sum"]).as_py()),
                float(pc.sum(table_5k["frag_gc_sum"]).as_py()),
                rtol=1e-5,
            )
            bands_5k = (
                table_5k["frag_bands"]
                .combine_chunks()
                .values.to_numpy()
                .reshape(-1, NUM_FRAG_BANDS)
            )
            bands_100k = (
                table_100k["frag_bands"]
                .combine_chunks()
                .values.to_numpy()
                .reshape(-1, NUM_FRAG_BANDS)
            )
            np.testing.assert_array_equal(bands_100k.sum(axis=0), bands_5k.sum(axis=0))

            # Non-nested coarsening redistributes count, nulls frag_bands, clears fragmentomics, logs warning
            with self.assertLogs("ccnv.features.rebinning", level=logging.WARNING):
                table_12k = rebin_sample_sketch(table_5k, 12_000)
            validate_sample_table(table_12k)
            assert int(pc.sum(table_12k["count"]).as_py()) == int(
                pc.sum(table_5k["count"]).as_py()
            )
            assert table_12k["frag_bands"].null_count == table_12k.num_rows
            assert int(pc.sum(table_12k["paired_count"]).as_py()) == 0
            assert FRAGMENTOMICS_METADATA_KEY not in (table_12k.schema.metadata or {})

            # Upsampling rejected with BinCompatibilityError
            with self.assertRaises(BinCompatibilityError):
                rebin_sample_sketch(table_5k, 1_000)

            # Single-end extraction emits null frag_bands and silent non-nested coarsening when ep_count==0
            se_bam = tmp / "se.bam"
            _write_single_end_bam(se_bam, [("chr1", 25_000)])
            se_table = extract_sample_table(
                ExtractConfig(alignment=se_bam, prefix=tmp / "se", bin_size=5_000)
            )
            assert se_table["frag_bands"].null_count == se_table.num_rows
            se_zero_ep = se_table.set_column(
                se_table.schema.get_field_index("endpoint_count"),
                "endpoint_count",
                pa.array(np.zeros(se_table.num_rows, dtype=np.uint32)),
            )
            with self.assertNoLogs("ccnv.features.rebinning", level=logging.WARNING):
                se_12k = rebin_sample_sketch(se_zero_ep, 12_000)
            assert int(pc.sum(se_12k["count"]).as_py()) == int(
                pc.sum(se_zero_ep["count"]).as_py()
            )

    def test_legacy_v1_table_rejected_with_version_error(self) -> None:
        legacy_meta = encode_metadata(
            {
                "artifact_kind": "sample",
                "schema_version": "1.0.0",
                "coordinate_system": COORDINATE_SYSTEM,
                "contig_lengths": {"chr1": 10_000},
                "bin_size": 5_000,
                "source_identity": {"path": "s.bam", "format": "bam"},
                "reference_identity": {"n_contigs": 1, "contig_order": ["chr1"]},
                "qc": {},
                "filters": asdict(ReadFilters()),
            }
        )
        legacy_v1_table = pa.table(
            {
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 5_000], type=pa.int64()),
                "end": pa.array([5_000, 10_000], type=pa.int64()),
                "count": pa.array([10, 12], type=pa.uint64()),
                "fragment_lengths": pa.array(
                    [[166], [170]], type=pa.list_(pa.uint16())
                ),
                "endpoint_count": pa.array([20, 24], type=pa.uint32()),
                "frag_gc_sum": pa.array([4.2, 5.1], type=pa.float32()),
            },
            metadata=legacy_meta,
        )
        with self.assertRaises(SchemaVersionError):
            validate_sample_table(legacy_v1_table)

        legacy_ref_meta = encode_metadata(
            {
                "artifact_kind": "reference",
                "schema_version": "1.0.0",
                "coordinate_system": COORDINATE_SYSTEM,
                "contig_lengths": {"chr1": 5_000},
                "sample_identities": ["s1"],
                "model": {"bin_size": 5_000, "branches": ["autosomal"]},
            }
        )
        legacy_ref_table = pa.table({"branch": ["autosomal"]}, metadata=legacy_ref_meta)
        with self.assertRaises(SchemaVersionError):
            validate_reference_table(legacy_ref_table)

    def test_reference_schema_v2_purged_prenatal_and_fast_vectorized_validation(
        self,
    ) -> None:
        # Verify dead mlp_*, olm_skew_*, preface_* fields are purged from PrenatalModelMetadata
        prenatal_fields = set(PrenatalModelMetadata.__dataclass_fields__.keys())
        for dead_prefix in ("mlp_", "olm_skew_", "preface_"):
            assert not any(f.startswith(dead_prefix) for f in prenatal_fields)
        assert not hasattr(PrenatalModelMetadata, "preface_std")
        assert not hasattr(PrenatalModelMetadata, "preface_logit_mode")

        # Validate rank-0 Parquet roundtrip and rank-4 vectorized validation
        ref_r0 = _make_valid_reference_table(20, 10, pca_rank=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_pq = write_reference_parquet(ref_r0, Path(tmpdir) / "ref_r0.ccnv")
            reloaded_r0, meta_r0 = read_reference_parquet(ref_pq, check_rows=True)
            assert meta_r0.bin_size == 5_000
            assert reloaded_r0.num_rows == 30

        ref_large = _make_valid_reference_table(50_000, 10_000, pca_rank=4)
        validate_reference_table(ref_large)  # warm up Arrow compute kernels
        t0 = perf_counter()
        ref_meta = validate_reference_table(ref_large, check_rows=True)
        elapsed_ms = (perf_counter() - t0) * 1_000.0
        assert ref_meta.branches == (BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX)
        assert elapsed_ms < 25.0, (
            f"expected fast vectorized validation, took {elapsed_ms:.2f} ms"
        )

        # Verify gap within run rejected
        bad_starts = ref_r0["start"].to_numpy().copy()
        bad_starts[5] += 5_000
        bad_gap = ref_r0.set_column(
            ref_r0.schema.get_field_index("start"),
            "start",
            pa.array(bad_starts, type=pa.uint32()),
        )
        with self.assertRaises(SchemaError):
            validate_reference_table(bad_gap, check_rows=True)
        # check_rows=False skips row scan
        validate_reference_table(bad_gap, check_rows=False)

        # Verify non-contiguous (branch, contig) run rejected
        non_contig_b = np.array([0] * 10 + [1] * 10 + [0] * 10, dtype=np.int8)
        bad_runs = ref_r0.set_column(
            0,
            "branch",
            pa.DictionaryArray.from_arrays(
                non_contig_b, [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX]
            ),
        )
        with self.assertRaises(SchemaError):
            validate_reference_table(bad_runs, check_rows=True)

    def test_dictionary_array_duplicate_vocab_and_null_dictionary_rejected(
        self,
    ) -> None:
        base_ref = _make_valid_reference_table(2, 0, pca_rank=1)

        # Duplicate contig vocabulary ["chr1", "chr1"] hiding duplicate [0, 5000) intervals
        dup_contig_col = pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int16()),
            pa.array(["chr1", "chr1"], type=pa.string()),
        )
        dup_contig_table = (
            base_ref.set_column(1, "contig", dup_contig_col)
            .set_column(2, "start", pa.array([0, 0], type=pa.uint32()))
            .set_column(3, "end", pa.array([5_000, 5_000], type=pa.uint32()))
        )
        with self.assertRaises(SchemaError):
            validate_reference_table(dup_contig_table, check_rows=True)

        # Duplicate branch vocabulary ["autosomal", "female_sex", "autosomal"] hiding non-contiguous branch runs
        ref_3 = _make_valid_reference_table(2, 1, pca_rank=1)
        dup_branch_col = pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 2], type=pa.int8()),
            pa.array(
                [BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_AUTOSOMAL],
                type=pa.string(),
            ),
        )
        dup_branch_table = (
            ref_3.set_column(0, "branch", dup_branch_col)
            .set_column(
                1,
                "contig",
                pa.DictionaryArray.from_arrays(
                    pa.array([0, 1, 0], type=pa.int16()), ["chr1", "chrX"]
                ),
            )
            .set_column(2, "start", pa.array([0, 0, 5_000], type=pa.uint32()))
            .set_column(3, "end", pa.array([5_000, 5_000, 10_000], type=pa.uint32()))
        )
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            validate_reference_table(dup_branch_table, check_rows=True)

        # Null entry inside DictionaryArray.dictionary must be rejected with SchemaError
        null_branch_dict = pa.DictionaryArray.from_arrays(
            pa.array([0, 0], type=pa.int8()),
            pa.array([BRANCH_AUTOSOMAL, None], type=pa.string()),
        )
        with self.assertRaisesRegex(SchemaError, "dictionary"):
            validate_reference_table(
                base_ref.set_column(0, "branch", null_branch_dict), check_rows=True
            )

        null_contig_dict = pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int16()),
            pa.array(["chr1", None], type=pa.string()),
        )
        with self.assertRaisesRegex(SchemaError, "dictionary"):
            validate_reference_table(
                base_ref.set_column(1, "contig", null_contig_dict), check_rows=True
            )

    def test_fixed_size_list_inner_nulls_rejected_on_sample_and_reference(
        self,
    ) -> None:
        sample_meta = metadata_for(
            ArtifactKind.SAMPLE,
            contig_lengths={"chr1": 10_000},
            values={
                "bin_size": 5_000,
                "source_identity": {"path": "s.bam"},
                "reference_identity": {"n_contigs": 1, "contig_order": ["chr1"]},
                "qc": {},
                "filters": asdict(ReadFilters()),
            },
        )
        bad_bands = pa.array(
            [[1, None, *([0] * (NUM_FRAG_BANDS - 2))], [0] * NUM_FRAG_BANDS],
            type=pa.list_(pa.uint32(), NUM_FRAG_BANDS),
        )
        assert bad_bands.null_count == 0
        bad_sample = pa.table(
            {
                "count": pa.array([5, 5], type=pa.uint32()),
                "paired_count": pa.array([5, 5], type=pa.uint32()),
                "frag_len_sum": pa.array([800, 800], type=pa.uint64()),
                "short_frag_count": pa.array([1, 1], type=pa.uint32()),
                "endpoint_count": pa.array([10, 10], type=pa.uint32()),
                "frag_gc_sum": pa.array([2.1, 2.1], type=pa.float32()),
                "frag_bands": bad_bands,
            },
            schema=SAMPLE_SCHEMA.with_metadata(sample_meta),
        )
        with self.assertRaisesRegex(
            SchemaError, "column 'frag_bands' contains null list elements"
        ):
            validate_sample_table(bad_sample)
        with self.assertRaisesRegex(
            SchemaError, "column 'frag_bands' contains null list elements"
        ):
            rebin_sample_sketch(bad_sample, 10_000)

        # Mixed row-level null + inner element null in frag_bands
        mixed_bands = pa.array(
            [None, [1, None, *([0] * (NUM_FRAG_BANDS - 2))]],
            type=pa.list_(pa.uint32(), NUM_FRAG_BANDS),
        )
        mixed_sample = bad_sample.set_column(6, "frag_bands", mixed_bands)
        with self.assertRaisesRegex(
            SchemaError, "column 'frag_bands' contains null list elements"
        ):
            validate_sample_table(mixed_sample)

        # Valid row-level null mixed with clean list row coarsens cleanly without NaN cast
        clean_row_null = pa.array(
            [None, [2] * NUM_FRAG_BANDS],
            type=pa.list_(pa.uint32(), NUM_FRAG_BANDS),
        )
        valid_mixed = bad_sample.set_column(6, "frag_bands", clean_row_null)
        coarsened = rebin_sample_sketch(valid_mixed, 10_000)
        assert coarsened["frag_bands"][0].as_py() == [2] * NUM_FRAG_BANDS

        # Reference FixedSizeList inner nulls (proto_ids and nullable gc_quantiles)
        base_ref = _make_valid_reference_table(2, 0, pca_rank=2)
        bad_proto = pa.array(
            [[0, None, 1, 2], [0, 1, 2, 3]],
            type=pa.list_(pa.uint16(), NUM_PROTOTYPES_PER_BIN),
        )
        with self.assertRaisesRegex(
            SchemaError, "column 'proto_ids' contains null list elements"
        ):
            validate_reference_table(
                base_ref.set_column(
                    base_ref.schema.get_field_index("proto_ids"),
                    "proto_ids",
                    bad_proto,
                )
            )

        bad_gc_q = pa.array(
            [None, [0.1, None, *([0.1] * (NUM_GC_QUANTILES - 2))]],
            type=pa.list_(pa.float32(), NUM_GC_QUANTILES),
        )
        with self.assertRaisesRegex(
            SchemaError, "column 'gc_quantiles' contains null list elements"
        ):
            validate_reference_table(
                base_ref.set_column(
                    base_ref.schema.get_field_index("gc_quantiles"),
                    "gc_quantiles",
                    bad_gc_q,
                )
            )

    def test_metadata_schema_error_boundary_and_sample_metadata_post_init(
        self,
    ) -> None:
        # 1. SampleMetadata.__post_init__ invariants raise SchemaError (never ZeroDivisionError)
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=0,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=True,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=5_000,
                contig_lengths={"chr1": 0},
                contig_order=("chr1",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )
        with self.assertRaises(SchemaError):
            SampleMetadata(
                bin_size=5_000,
                contig_lengths={"chr1": 10_000},
                contig_order=("chr2",),
                source_identity={},
                reference_identity={},
                qc={},
                filters={},
            )

        # 2. Non-mapping or malformed ccnv.fragmentomics raises SchemaError everywhere
        base_s_meta = decode_metadata(
            metadata_for(
                ArtifactKind.SAMPLE,
                contig_lengths={"chr1": 5_000},
                values={
                    "bin_size": 5_000,
                    "source_identity": {},
                    "reference_identity": {},
                    "qc": {},
                    "filters": asdict(ReadFilters()),
                },
            )
        )
        for bad_frag in (123, ["not", "mapping"], {"paired_end": True}):
            bad_dict = {**base_s_meta, "fragmentomics": bad_frag}
            with self.assertRaises(SchemaError):
                validate_metadata(bad_dict, ArtifactKind.SAMPLE)
            with self.assertRaises(SchemaError):
                validate_schema(
                    SAMPLE_SCHEMA.with_metadata(encode_metadata(bad_dict)),
                    ArtifactKind.SAMPLE,
                )
            with self.assertRaises(SchemaError):
                SampleMetadata.from_dict(bad_dict)

        # 3. Malformed field types in SampleMetadata / ReferenceMetadata raise SchemaError (not TypeError/KeyError/ValueError)
        with self.assertRaises(SchemaError):
            SampleMetadata.from_dict({})
        with self.assertRaises(SchemaError):
            SampleMetadata.from_dict({**base_s_meta, "bin_size": "five_kb"})

        base_ref = _make_valid_reference_table(2, 0, pca_rank=1)
        ref_dict = decode_metadata(base_ref.schema.metadata)
        model_obj = ref_dict["model"]
        assert isinstance(model_obj, dict)
        for bad_field, bad_val in (
            ("pca_components", "five"),
            ("min_mappability", "high"),
            ("nearest_bins", "many"),
        ):
            corrupt_ref_dict = {
                **ref_dict,
                "model": {**model_obj, bad_field: bad_val},
            }
            corrupt_table = base_ref.replace_schema_metadata(
                encode_metadata(corrupt_ref_dict)
            )
            with self.assertRaises(SchemaError):
                ReferenceMetadata.from_dict(corrupt_ref_dict)
            with self.assertRaises(SchemaError):
                ReferenceMetadata.from_table(corrupt_table)
            with self.assertRaises(SchemaError):
                validate_reference_table(corrupt_table)


if __name__ == "__main__":
    unittest.main()
