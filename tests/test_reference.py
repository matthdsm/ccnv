import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from published_fixtures import (
    CHR21_LENGTH,
    PUBLISHED_BIN_SIZE,
    PUBLISHED_CRAM,
    PUBLISHED_CRAM2,
    PUBLISHED_CRAM2_URL,
    PUBLISHED_CRAM_URL,
    ensure_published_fasta,
    ensure_published_file,
)
from sklearn.neighbors import NearestNeighbors
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.convert import read_sample_parquet
from ccnv.models import PredictConfig, ReadFilters, ReferenceConfig, parquet_path
from ccnv.predict import run as predict
from ccnv.reference import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    DEFAULT_SEX_CONTIGS,
    ReferenceError,
    build_reference_table,
    cohort_from_fasta,
    fit_pca,
    infer_sample_sexes,
    load_cohort,
    mask_low_coverage,
    normalize_depth,
    run,
    select_nearest_bins,
    validate_cohort,
)
from ccnv.schemas import (
    REFERENCE_SCHEMA,
    SAMPLE_SCHEMA,
    ArtifactKind,
    SchemaError,
    decode_metadata,
    metadata_for,
    validate_table,
)


def _sample_table(
    counts: list[int],
    *,
    contigs: list[str] | None = None,
    starts: list[int] | None = None,
    ends: list[int] | None = None,
    contig_lengths: dict[str, int] | None = None,
    bin_size: int = 100,
    identity: object | None = None,
) -> pa.Table:
    n_bins = len(counts)
    if starts is None:
        starts = [i * bin_size for i in range(n_bins)]
    if ends is None:
        ends = [start + bin_size for start in starts]
    if contigs is None:
        contigs = ["chr1"] * n_bins
        if contig_lengths is None:
            contig_lengths = {"chr1": ends[-1]}
    elif contig_lengths is None:
        contig_lengths = {}
        for name in dict.fromkeys(contigs):
            contig_lengths[name] = max(
                end for contig, end in zip(contigs, ends, strict=True) if contig == name
            )
    metadata = metadata_for(
        ArtifactKind.SAMPLE,
        contig_lengths=contig_lengths,
        values={
            "bin_size": bin_size,
            "source_identity": identity if identity is not None else {"id": "sample"},
            "reference_identity": {"name": "test"},
            "qc": {},
            "filters": asdict(ReadFilters()),
        },
    )
    return pa.table(
        {
            "contig": contigs,
            "start": pa.array(starts, type=pa.int64()),
            "end": pa.array(ends, type=pa.int64()),
            "count": pa.array(counts, type=pa.uint64()),
        },
        schema=SAMPLE_SCHEMA.with_metadata(metadata),
    )


def _write_sample(directory: Path, name: str, table: pa.Table) -> Path:
    path = directory / name
    pq.write_table(table, path)
    return path


class TestReferenceSchemaRejection(unittest.TestCase):
    def test_load_cohort_rejects_sample_schema_violations(self) -> None:
        table = pa.table(
            {
                "contig": ["chr1"],
                "start": pa.array([0], type=pa.int64()),
                "end": pa.array([10], type=pa.int64()),
                "count": pa.array([1], type=pa.uint64()),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_sample(Path(tmpdir), "bad.ccnv", table)
            with self.assertRaisesRegex(ReferenceError, "fields do not match"):
                load_cohort((path,))

    def test_validate_table_rejects_duplicate_branch_bins(self) -> None:
        metadata = metadata_for(
            ArtifactKind.REFERENCE,
            contig_lengths={"chr1": 10},
            values={"model": {}, "sample_identities": []},
        )
        table = pa.table(
            {
                "branch": [BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
                "contig": ["chr1", "chr1"],
                "start": pa.array([0, 0], type=pa.int64()),
                "end": pa.array([10, 10], type=pa.int64()),
                "masked": [False, False],
                "pca_mean": pa.array([0.0, 0.0], type=pa.float64()),
                "pca_loadings": pa.array([[0.1], [0.1]], type=pa.list_(pa.float64())),
                "nearest_ref_indices": pa.array([[], []], type=pa.list_(pa.int32())),
                "nearest_ref_distances": pa.array(
                    [[], []], type=pa.list_(pa.float64())
                ),
                "null_ratios": pa.array([[], []], type=pa.list_(pa.float64())),
            },
            schema=REFERENCE_SCHEMA.with_metadata(metadata),
        )
        with self.assertRaisesRegex(SchemaError, "duplicates bin identity"):
            validate_table(table, ArtifactKind.REFERENCE)


class TestCohortCompatibility(unittest.TestCase):
    def test_rejects_bin_size_mismatch(self) -> None:
        left = _sample_table([10, 10], bin_size=100)
        right = _sample_table([5, 5, 5, 5], bin_size=50)
        with self.assertRaisesRegex(ReferenceError, "bin_size"):
            validate_cohort((left, right))

    def test_rejects_coordinate_mismatch(self) -> None:
        left = _sample_table([10, 10], ends=[100, 200], contig_lengths={"chr1": 200})
        right = _sample_table(
            [10, 10],
            starts=[0, 80],
            ends=[80, 200],
            contig_lengths={"chr1": 200},
        )
        with self.assertRaisesRegex(ReferenceError, "bin coordinates"):
            validate_cohort((left, right))

    def test_mixed_native_bin_sizes_succeed_when_target_bin_size_set(self) -> None:
        left = _sample_table([10, 10], bin_size=100)
        right = _sample_table([5, 5, 5, 5], bin_size=50)
        cohort = validate_cohort((left, right), target_bin_size=100)
        self.assertEqual(cohort.bin_size, 100)
        self.assertEqual(cohort.counts.shape, (2, 2))
        np.testing.assert_allclose(cohort.counts[:, 0], [10.0, 10.0])
        np.testing.assert_allclose(cohort.counts[:, 1], [10.0, 10.0])

    def test_mismatched_contig_lengths_fail_even_with_target_bin_size(self) -> None:
        left = _sample_table([10, 10], contig_lengths={"chr1": 200}, bin_size=100)
        right = _sample_table([10], contig_lengths={"chr1": 100}, bin_size=100)
        with self.assertRaisesRegex(ReferenceError, "contig_lengths do not match"):
            validate_cohort((left, right), target_bin_size=100)

    def test_accepts_matching_nonhuman_layout(self) -> None:
        contigs = ["scaffold-A", "scaffold-A", "lgX"]
        starts = [0, 100, 0]
        ends = [100, 200, 100]
        lengths = {"scaffold-A": 200, "lgX": 100}
        tables = [
            _sample_table(
                [8, 9, 2],
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths=lengths,
                identity={"id": f"s{index}"},
            )
            for index in range(3)
        ]
        cohort = validate_cohort(tuple(tables))
        self.assertEqual(cohort.counts.shape, (3, 3))
        self.assertEqual(cohort.contigs, ("scaffold-A", "scaffold-A", "lgX"))


class TestMaskingAndPca(unittest.TestCase):
    def test_mask_drops_empty_and_low_coverage_bins(self) -> None:
        counts = np.array(
            [
                [100.0, 100.0, 100.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        normalized = normalize_depth(counts)
        mask = mask_low_coverage(normalized)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_pca_is_deterministic_and_matches_unmasked_bins(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.random((12, 6)) + 0.2
        first = fit_pca(data, n_components=3)
        second = fit_pca(data, n_components=3)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        np.testing.assert_allclose(first[2], second[2])
        self.assertEqual(first[1].shape, (12,))
        self.assertEqual(first[2].shape, (3, 12))


class TestNearestBinSelection(unittest.TestCase):
    def test_selects_closest_off_contig_bins(self) -> None:
        profiles = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.05, 0.95, 0.0],
            ]
        )
        contig_ids = np.array([0, 0, 1, 1])
        indices, distances = select_nearest_bins(
            profiles, contig_ids, k=1, max_batch_memory_mb=1
        )
        self.assertEqual(indices[0], [2])
        self.assertEqual(indices[1], [3])
        self.assertEqual(indices[2], [0])
        self.assertEqual(indices[3], [1])
        self.assertTrue(all(len(row) == 1 for row in distances))

    def test_argpartition_matches_nearest_neighbors_on_small_matrix(self) -> None:
        rng = np.random.default_rng(1)
        profiles = rng.normal(size=(8, 5))
        contig_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        indices, distances = select_nearest_bins(
            profiles, contig_ids, k=2, max_batch_memory_mb=1
        )
        model = NearestNeighbors(n_neighbors=8, algorithm="brute", metric="euclidean")
        model.fit(profiles)
        brute_dist, brute_idx = model.kneighbors(profiles, return_distance=True)
        for query, neighbor_idx, neighbor_dist in zip(
            range(len(profiles)), brute_idx, brute_dist, strict=True
        ):
            expected = [
                (int(index), float(dist) ** 2)
                for index, dist in zip(neighbor_idx, neighbor_dist, strict=True)
                if contig_ids[index] != contig_ids[query]
            ][:2]
            self.assertEqual(indices[query], [index for index, _ in expected])
            np.testing.assert_allclose(
                distances[query],
                [dist for _, dist in expected],
                rtol=1e-6,
                atol=1e-8,
            )


class TestSexInference(unittest.TestCase):
    def test_infers_male_as_higher_signal_fraction(self) -> None:
        fractions = np.array([0.001, 0.0012, 0.0009, 0.02, 0.021, 0.019])
        result = infer_sample_sexes(fractions, min_group_size=2)
        self.assertTrue(result.passed)
        self.assertEqual(
            result.labels,
            ("female", "female", "female", "male", "male", "male"),
        )
        self.assertIsNotNone(result.cutoff)
        self.assertGreater(result.cutoff or 0.0, 0.0012)
        self.assertLess(result.cutoff or 1.0, 0.019)

    def test_rejects_overlapping_fractions(self) -> None:
        fractions = np.linspace(0.01, 0.012, 8)
        result = infer_sample_sexes(fractions, min_group_size=3)
        self.assertFalse(result.passed)
        self.assertIsNone(result.labels)

    def test_cutoff_fraction_classifies_male_above_threshold(self) -> None:
        fractions = np.array([0.001, 0.002, 0.02, 0.021])
        result = infer_sample_sexes(fractions, min_group_size=2, cutoff=0.01)
        self.assertTrue(result.passed)
        self.assertEqual(result.labels, ("female", "female", "male", "male"))
        self.assertEqual(result.cutoff, 0.01)
        self.assertEqual(result.reason, "cutoff")

    def test_defaults_select_human_sex_contigs_and_chrY_signal(self) -> None:
        females = [[40, 40, 2, 1], [38, 42, 1, 3], [41, 39, 3, 2]]
        males = [[40, 40, 20, 18], [39, 41, 18, 22], [42, 38, 22, 19]]
        contigs = ["chr1", "chr1", "chrY", "chrY"]
        starts = [0, 100, 0, 100]
        ends = [100, 200, 100, 200]
        lengths = {"chr1": 200, "chrY": 200}
        tables = [
            _sample_table(
                counts,
                contigs=contigs,
                starts=starts,
                ends=ends,
                contig_lengths=lengths,
                identity={"id": f"s{index}"},
            )
            for index, counts in enumerate(females + males)
        ]
        cohort = validate_cohort(tuple(tables))
        table = build_reference_table(
            cohort,
            ReferenceConfig(
                samples=tuple(Path(f"{index}.ccnv") for index in range(6)),
                prefix=Path("unused"),
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
                min_sex_group_size=3,
            ),
        )
        validate_table(table, ArtifactKind.REFERENCE)
        branches = set(table.column("branch").to_pylist())
        self.assertEqual(
            branches,
            {BRANCH_AUTOSOMAL, BRANCH_FEMALE_SEX, BRANCH_MALE_SEX},
        )
        model = decode_metadata(table.schema.metadata)["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["sex_source"], "inferred")
        self.assertEqual(model["autosomal_contigs"], ["chr1"])
        self.assertEqual(model["sex_contigs"], ["chrY"])
        self.assertEqual(model["sex_signal_contig"], "chrY")

    def test_defaults_prefer_chrY_signal_and_include_chrX(self) -> None:
        females = [[40, 2, 1], [38, 1, 3], [41, 3, 2]]
        males = [[40, 20, 18], [39, 18, 22], [42, 22, 19]]
        tables = [
            _sample_table(
                counts,
                contigs=["chr1", "chrX", "chrY"],
                starts=[0, 0, 0],
                ends=[100, 100, 100],
                contig_lengths={"chr1": 100, "chrX": 100, "chrY": 100},
                identity={"id": f"s{index}"},
            )
            for index, counts in enumerate(females + males)
        ]
        table = build_reference_table(
            validate_cohort(tuple(tables)),
            ReferenceConfig(
                samples=tuple(Path(f"{index}.ccnv") for index in range(6)),
                prefix=Path("unused"),
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
                min_sex_group_size=3,
            ),
        )
        model = decode_metadata(table.schema.metadata)["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["autosomal_contigs"], ["chr1"])
        self.assertEqual(model["sex_contigs"], ["chrX", "chrY"])
        self.assertEqual(model["sex_signal_contig"], "chrY")

    def test_defaults_select_unprefixed_x_and_y(self) -> None:
        tables = [
            _sample_table(
                [40, 2, 1],
                contigs=["1", "X", "Y"],
                starts=[0, 0, 0],
                ends=[100, 100, 100],
                contig_lengths={"1": 100, "X": 100, "Y": 100},
                identity={"id": f"s{index}"},
            )
            for index in range(4)
        ]
        table = build_reference_table(
            validate_cohort(tuple(tables)),
            ReferenceConfig(
                samples=tuple(Path(f"{index}.ccnv") for index in range(4)),
                prefix=Path("unused"),
                sex_contigs=DEFAULT_SEX_CONTIGS,
                sex_signal_contig="chrY",
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
                min_sex_group_size=2,
            ),
        )
        model = decode_metadata(table.schema.metadata)["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["autosomal_contigs"], ["1"])
        self.assertEqual(model["sex_contigs"], ["X", "Y"])
        self.assertEqual(model["sex_signal_contig"], "Y")

    def test_defaults_skip_missing_human_sex_names(self) -> None:
        tables = [
            _sample_table(
                [40, 40],
                contigs=["chr1", "scaffold"],
                starts=[0, 0],
                ends=[100, 100],
                contig_lengths={"chr1": 100, "scaffold": 100},
                identity={"id": f"s{index}"},
            )
            for index in range(4)
        ]
        table = build_reference_table(
            validate_cohort(tuple(tables)),
            ReferenceConfig(
                samples=tuple(Path(f"{index}.ccnv") for index in range(4)),
                prefix=Path("unused"),
                sex_contigs=DEFAULT_SEX_CONTIGS,
                sex_signal_contig="chrY",
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
            ),
        )
        model = decode_metadata(table.schema.metadata)["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["autosomal_contigs"], ["chr1", "scaffold"])
        self.assertEqual(model["sex_contigs"], [])
        self.assertIsNone(model["sex_signal_contig"])
        self.assertEqual(set(table.column("branch").to_pylist()), {BRANCH_AUTOSOMAL})

    def test_unknown_sex_contig_is_rejected(self) -> None:
        tables = [
            _sample_table(
                [40, 40],
                contigs=["chr1", "chrY"],
                starts=[0, 0],
                ends=[100, 100],
                contig_lengths={"chr1": 100, "chrY": 100},
                identity={"id": f"s{index}"},
            )
            for index in range(4)
        ]
        cohort = validate_cohort(tuple(tables))
        config = ReferenceConfig(
            samples=tuple(Path(f"{index}.ccnv") for index in range(4)),
            prefix=Path("unused"),
            sex_contigs=("chrZ",),
            pca_components=2,
            nearest_bins=1,
            min_cohort_size=4,
        )
        with self.assertRaisesRegex(ReferenceError, "unknown sex contig"):
            build_reference_table(cohort, config)

    def test_unknown_sex_signal_contig_is_rejected(self) -> None:
        tables = [
            _sample_table(
                [40, 40],
                contigs=["chr1", "chrY"],
                starts=[0, 0],
                ends=[100, 100],
                contig_lengths={"chr1": 100, "chrY": 100},
                identity={"id": f"s{index}"},
            )
            for index in range(4)
        ]
        config = ReferenceConfig(
            samples=tuple(Path(f"{index}.ccnv") for index in range(4)),
            prefix=Path("unused"),
            sex_signal_contig="chrZ",
            pca_components=2,
            nearest_bins=1,
            min_cohort_size=4,
        )
        with self.assertRaisesRegex(ReferenceError, "unknown sex-signal contig"):
            build_reference_table(validate_cohort(tuple(tables)), config)

    def test_cutoff_rejects_values_outside_unit_interval(self) -> None:
        fractions = np.array([0.001, 0.02, 0.021, 0.022])
        with self.assertRaisesRegex(ReferenceError, r"\[0, 1\]"):
            infer_sample_sexes(fractions, min_group_size=2, cutoff=1.5)
        with self.assertRaisesRegex(ReferenceError, r"\[0, 1\]"):
            infer_sample_sexes(fractions, min_group_size=2, cutoff=float("nan"))

    def test_cutoff_builds_sex_branches_without_gap_detection(self) -> None:
        counts = [[50, 1], [50, 2], [50, 3], [50, 20], [50, 21], [50, 22]]
        tables = [
            _sample_table(
                values,
                contigs=["chr1", "chrY"],
                starts=[0, 0],
                ends=[100, 100],
                contig_lengths={"chr1": 100, "chrY": 100},
                identity={"id": f"s{index}"},
            )
            for index, values in enumerate(counts)
        ]
        table = build_reference_table(
            validate_cohort(tuple(tables)),
            ReferenceConfig(
                samples=tuple(Path(f"{index}.ccnv") for index in range(6)),
                prefix=Path("unused"),
                sex_cutoff=0.1,
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
                min_sex_group_size=3,
            ),
        )
        model = decode_metadata(table.schema.metadata)["model"]
        assert isinstance(model, dict)
        self.assertEqual(model["sex_cutoff"], 0.1)
        self.assertEqual(
            model["sample_sexes"],
            ["female", "female", "female", "male", "male", "male"],
        )

    def test_inferred_sex_is_deterministic(self) -> None:
        female_counts = [[50, 50, 1, 2] for _ in range(3)]
        male_counts = [[50, 50, 20, 18] for _ in range(3)]
        female_counts[1] = [48, 52, 2, 1]
        female_counts[2] = [51, 49, 1, 3]
        male_counts[1] = [47, 53, 22, 16]
        male_counts[2] = [52, 48, 18, 21]
        contigs = ["chr1", "chr1", "chrY", "chrY"]
        starts = [0, 100, 0, 100]
        ends = [100, 200, 100, 200]
        lengths = {"chr1": 200, "chrY": 200}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = [
                _write_sample(
                    root,
                    f"{index}.ccnv",
                    _sample_table(
                        counts,
                        contigs=contigs,
                        starts=starts,
                        ends=ends,
                        contig_lengths=lengths,
                        identity={"id": f"s{index}"},
                    ),
                )
                for index, counts in enumerate(female_counts + male_counts)
            ]
            config = ReferenceConfig(
                samples=tuple(paths),
                prefix=root / "ref",
                sex_contigs=("chrY",),
                sex_signal_contig="chrY",
                pca_components=2,
                nearest_bins=1,
                min_cohort_size=4,
                min_sex_group_size=3,
            )
            run(config)
            run(replace(config, prefix=root / "ref2"))
            first = pq.read_table(root / "ref.ccnv")
            second = pq.read_table(root / "ref2.ccnv")
            validate_table(first, ArtifactKind.REFERENCE)
            self.assertEqual(
                decode_metadata(first.schema.metadata)["model"],
                decode_metadata(second.schema.metadata)["model"],
            )
            self.assertEqual(first.to_pydict(), second.to_pydict())
            model = decode_metadata(first.schema.metadata)["model"]
            assert isinstance(model, dict)
            self.assertEqual(model["sex_source"], "inferred")
            self.assertEqual(
                model["sample_sexes"],
                ["female", "female", "female", "male", "male", "male"],
            )


class TestReferencePublishedCrams(unittest.TestCase):
    def test_cli_builds_reference_from_two_published_crams(self) -> None:
        fasta = ensure_published_fasta()
        crams = (
            ensure_published_file(PUBLISHED_CRAM, PUBLISHED_CRAM_URL),
            ensure_published_file(PUBLISHED_CRAM2, PUBLISHED_CRAM2_URL),
        )
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_paths: list[Path] = []
            for index, cram in enumerate(crams, start=1):
                prefix = root / f"sample{index}"
                result = runner.invoke(
                    app,
                    [
                        "convert",
                        str(cram),
                        "--fasta",
                        str(fasta),
                        "--prefix",
                        str(prefix),
                        "--bin-size",
                        str(PUBLISHED_BIN_SIZE),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                sample_paths.append(parquet_path(prefix))

            samples = [read_sample_parquet(path) for path in sample_paths]
            self.assertEqual(samples[0].num_rows, samples[1].num_rows)
            self.assertEqual(
                set(samples[0].column("contig").to_pylist()),
                {"chr21"},
            )

            reference_prefix = root / "reference"
            result = runner.invoke(
                app,
                [
                    "reference",
                    *(str(path) for path in sample_paths),
                    "--prefix",
                    str(reference_prefix),
                    "--min-cohort-size",
                    "2",
                    "--pca-components",
                    "1",
                    "--nearest-bins",
                    "16",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)

            table = pq.read_table(parquet_path(reference_prefix))
            validate_table(table, ArtifactKind.REFERENCE)
            metadata = decode_metadata(table.schema.metadata)
            model = metadata["model"]
            assert isinstance(model, dict)
            self.assertEqual(table.num_rows, samples[0].num_rows)
            self.assertEqual(set(table.column("contig").to_pylist()), {"chr21"})
            self.assertEqual(
                set(table.column("branch").to_pylist()), {BRANCH_AUTOSOMAL}
            )
            self.assertEqual(table.column("end").to_pylist()[-1], CHR21_LENGTH)
            self.assertEqual(metadata["contig_lengths"], {"chr21": CHR21_LENGTH})
            self.assertEqual(model["bin_size"], PUBLISHED_BIN_SIZE)
            self.assertEqual(model["branches"], [BRANCH_AUTOSOMAL])
            self.assertEqual(model["autosomal_contigs"], ["chr21"])
            self.assertEqual(len(metadata["sample_identities"]), 2)
            unmasked = sum(1 for masked in table.column("masked") if not masked)
            self.assertGreater(unmasked, 0)


class TestFlatFastaReference(unittest.TestCase):
    def test_fasta_builds_uniform_bins_and_includes_sex_as_autosomal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(
                Path(tmpdir) / "genome.fa",
                (("chr1", 250), ("chr2", 150), ("chrX", 100)),
            )
            cohort = cohort_from_fasta(fasta, 100, n_samples=2)
            self.assertEqual(cohort.bin_size, 100)
            self.assertEqual(
                cohort.contigs, ("chr1", "chr1", "chr1", "chr2", "chr2", "chrX")
            )
            np.testing.assert_array_equal(cohort.starts, [0, 100, 200, 0, 100, 0])
            np.testing.assert_array_equal(cohort.ends, [100, 200, 250, 100, 150, 100])
            np.testing.assert_array_equal(
                cohort.counts[:, 0], [100, 100, 50, 100, 50, 100]
            )
            np.testing.assert_array_equal(cohort.counts[:, 0], cohort.counts[:, 1])

            prefix = Path(tmpdir) / "ref"
            run(
                ReferenceConfig(
                    samples=(),
                    prefix=prefix,
                    fasta=fasta,
                    bin_size=100,
                    pca_components=1,
                    nearest_bins=2,
                    min_cohort_size=2,
                )
            )
            table = pq.read_table(parquet_path(prefix))
            validate_table(table, ArtifactKind.REFERENCE)
            model = decode_metadata(table.schema.metadata)["model"]
            assert isinstance(model, dict)
            self.assertTrue(model["flat"])
            self.assertEqual(model["branches"], [BRANCH_AUTOSOMAL])
            self.assertEqual(model["autosomal_contigs"], ["chr1", "chr2", "chrX"])
            self.assertEqual(model["sex_contigs"], [])
            self.assertEqual(
                set(table.column("contig").to_pylist()), {"chr1", "chr2", "chrX"}
            )
            self.assertFalse(any(table.column("masked").to_pylist()))

    def test_flat_reference_lets_predict_run_without_control_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fasta = _write_tiny_fasta(
                root / "genome.fa",
                (("chr1", 250), ("chr2", 150)),
            )
            reference_prefix = root / "ref"
            run(
                ReferenceConfig(
                    samples=(),
                    prefix=reference_prefix,
                    fasta=fasta,
                    bin_size=100,
                    pca_components=1,
                    nearest_bins=2,
                    min_cohort_size=2,
                )
            )
            sample = _sample_table(
                [200, 200, 50, 100, 50],
                contigs=["chr1", "chr1", "chr1", "chr2", "chr2"],
                starts=[0, 100, 200, 0, 100],
                ends=[100, 200, 250, 100, 150],
                contig_lengths={"chr1": 250, "chr2": 150},
                bin_size=100,
            )
            sample_path = _write_sample(root, "sample.ccnv", sample)
            outputs = predict(
                PredictConfig(
                    sample=sample_path,
                    reference=parquet_path(reference_prefix),
                    prefix=root / "prediction",
                    plot=False,
                    min_reference_bins=1,
                )
            )
            self.assertTrue(outputs.bins_bed.is_file())
            self.assertTrue(outputs.segments_bed.is_file())
            self.assertEqual(len(outputs.bins), 1)
            self.assertGreater(len(outputs.bins[0].contigs), 0)
            self.assertGreater(int(np.count_nonzero(~outputs.bins[0].masked)), 0)

    def test_cli_builds_flat_reference_from_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(
                Path(tmpdir) / "genome.fa", (("chr1", 200), ("chr2", 200))
            )
            prefix = Path(tmpdir) / "ref"
            result = CliRunner().invoke(
                app,
                [
                    "reference",
                    str(fasta),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                    "--min-cohort-size",
                    "2",
                    "--pca-components",
                    "1",
                    "--nearest-bins",
                    "2",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            table = pq.read_table(parquet_path(prefix))
            model = decode_metadata(table.schema.metadata)["model"]
            assert isinstance(model, dict)
            self.assertTrue(model["flat"])
            self.assertEqual(table.num_rows, 4)

    def test_samples_and_fasta_together_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(Path(tmpdir) / "genome.fa", (("chr1", 100),))
            with self.assertRaisesRegex(ReferenceError, "not both"):
                run(
                    ReferenceConfig(
                        samples=(Path("sample.ccnv"),),
                        prefix=Path(tmpdir) / "ref",
                        fasta=fasta,
                        bin_size=100,
                    )
                )

    def test_neither_samples_nor_fasta_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReferenceError, "sample Parquet paths or a FASTA"):
            run(ReferenceConfig(samples=(), prefix=Path("ref")))


def _write_tiny_fasta(path: Path, records: tuple[tuple[str, int], ...]) -> Path:
    with path.open("w", encoding="ascii") as handle:
        for name, length in records:
            handle.write(f">{name}\n{'A' * length}\n")
    pysam.faidx(str(path))
    return path


if __name__ == "__main__":
    unittest.main()
