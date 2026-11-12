import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import yaml
from helpers import make_reference_table, make_sample_table
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
from typer.testing import CliRunner

from ccnv.cli import app
from ccnv.features.cohort import (
    cohort_from_fasta,
    load_and_assemble_cohort,
    mask_low_coverage_per_contig,
    validate_cohort,
)
from ccnv.features.sex import infer_sample_sexes
from ccnv.io.parquet import read_sample_buffers
from ccnv.io.tracks import read_fasta_contigs
from ccnv.models import (
    BRANCH_AUTOSOMAL,
    BRANCH_FEMALE_SEX,
    BRANCH_MALE_SEX,
    PredictConfig,
    ReferenceConfig,
    ReferenceError,
    SamplesheetEntry,
    SchemaError,
    decode_metadata,
    parquet_path,
    validate_reference_table,
)
from ccnv.predict import run as predict
from ccnv.reference import (
    build_reference_table,
    run,
)
from ccnv.stats.pca import fit_whitened_pca


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
    return make_sample_table(
        counts,
        contigs=contigs,
        starts=starts,
        ends=ends,
        contig_lengths=contig_lengths,
        bin_size=bin_size,
        identity=identity,
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
            with self.assertRaisesRegex(
                ReferenceError, "artifact_kind|fields do not match"
            ):
                load_and_assemble_cohort((SamplesheetEntry(filepath=path),))

    def test_validate_table_rejects_duplicate_branch_bins(self) -> None:
        table = make_reference_table(
            branches=[BRANCH_AUTOSOMAL, BRANCH_AUTOSOMAL],
            contigs=["chr1", "chr1"],
            starts=[0, 0],
            ends=[10, 10],
            contig_lengths={"chr1": 10},
            pca_loadings=[[0.1], [0.1]],
        )
        with self.assertRaisesRegex(
            SchemaError,
            "does not match previous bin end|duplicates|strictly monotonic|non-overlapping|contiguous",
        ):
            validate_reference_table(table)


class TestMaskingAndPca(unittest.TestCase):
    def test_mask_drops_empty_and_low_coverage_bins(self) -> None:
        counts = np.array(
            [
                [100.0, 100.0, 100.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        mask = mask_low_coverage_per_contig(
            counts,
            np.zeros(3, dtype=np.int16),
            np.ones(3, dtype=np.float32),
        )
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_pca_is_deterministic_and_matches_unmasked_bins(self) -> None:
        rng = np.random.default_rng(0)
        expected = np.full((12, 6), 100.0, dtype=np.float32)
        u = rng.normal(0.0, 0.25, size=(12, 2))
        v = rng.normal(0.0, 1.0, size=(2, 6))
        counts = rng.poisson(expected.astype(np.float64) * np.exp(u @ v)).astype(
            np.float32
        )
        usable = np.ones(12, dtype=bool)
        first = fit_whitened_pca(counts, expected, usable, max_components=3)
        second = fit_whitened_pca(counts, expected, usable, max_components=3)
        np.testing.assert_allclose(first.mean_rate, second.mean_rate)
        np.testing.assert_allclose(first.loadings, second.loadings)
        np.testing.assert_allclose(first.singular_values, second.singular_values)
        self.assertEqual(first.mean_rate.shape, (12,))
        self.assertGreaterEqual(first.loadings.shape[0], 1)
        self.assertLessEqual(first.loadings.shape[0], 3)
        self.assertEqual(first.loadings.shape[1], 12)


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
                entries=tuple(
                    SamplesheetEntry(filepath=Path(f"{index}.ccnv"))
                    for index in range(6)
                ),
                prefix=Path("unused"),
                pca_components=2,
                min_cohort_size=4,
                min_sex_group_size=3,
            ),
        )
        validate_reference_table(table)
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
                entries=tuple(
                    SamplesheetEntry(filepath=Path(f"{index}.ccnv"))
                    for index in range(6)
                ),
                prefix=Path("unused"),
                pca_components=2,
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
        females = [[40, 20, 1], [38, 21, 2]]
        males = [[40, 10, 18], [42, 11, 19]]
        tables = [
            _sample_table(
                counts,
                contigs=["1", "X", "Y"],
                starts=[0, 0, 0],
                ends=[100, 100, 100],
                contig_lengths={"1": 100, "X": 100, "Y": 100},
                identity={"id": f"s{index}"},
            )
            for index, counts in enumerate(females + males)
        ]
        table = build_reference_table(
            validate_cohort(tuple(tables)),
            ReferenceConfig(
                entries=tuple(
                    SamplesheetEntry(filepath=Path(f"{index}.ccnv"))
                    for index in range(4)
                ),
                prefix=Path("unused"),
                pca_components=2,
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
                entries=tuple(
                    SamplesheetEntry(filepath=Path(f"{index}.ccnv"))
                    for index in range(4)
                ),
                prefix=Path("unused"),
                pca_components=2,
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
            entries=tuple(
                SamplesheetEntry(filepath=Path(f"{index}.ccnv")) for index in range(4)
            ),
            prefix=Path("unused"),
            sex_contigs=("chrZ",),
            pca_components=2,
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
            entries=tuple(
                SamplesheetEntry(filepath=Path(f"{index}.ccnv")) for index in range(4)
            ),
            prefix=Path("unused"),
            sex_signal_contig="chrZ",
            pca_components=2,
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
                entries=tuple(
                    SamplesheetEntry(filepath=Path(f"{index}.ccnv"))
                    for index in range(6)
                ),
                prefix=Path("unused"),
                sex_cutoff=0.1,
                pca_components=2,
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
                entries=tuple(SamplesheetEntry(filepath=p) for p in paths),
                prefix=root / "ref",
                sex_contigs=("chrY",),
                sex_signal_contig="chrY",
                pca_components=2,
                min_cohort_size=4,
                min_sex_group_size=3,
            )
            run(config)
            run(replace(config, prefix=root / "ref2"))
            first = pq.read_table(root / "ref.ccnv")
            second = pq.read_table(root / "ref2.ccnv")
            validate_reference_table(first)
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

    def test_reference_table_schema_and_varying_cohort_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            contigs = ["chr1", "chr1", "chr2", "chr2"]
            starts = [0, 100, 0, 100]
            ends = [100, 200, 100, 200]
            lengths = {"chr1": 200, "chr2": 200}

            for cohort_size in (4, 7):
                paths = []
                for i in range(cohort_size):
                    counts = [10 + i, 12 - i % 3, 15 + i % 2, 9 + i]
                    table = _sample_table(
                        counts,
                        contigs=contigs,
                        starts=starts,
                        ends=ends,
                        contig_lengths=lengths,
                    )
                    p = _write_sample(root, f"sample_{cohort_size}_{i}.ccnv", table)
                    paths.append(p)

                ref_prefix = root / f"ref_{cohort_size}"
                config = ReferenceConfig(
                    entries=tuple(SamplesheetEntry(filepath=p) for p in paths),
                    prefix=ref_prefix,
                    pca_components=2,
                    min_cohort_size=4,
                )
                run(config)
                ref_path = root / f"ref_{cohort_size}.ccnv"
                table = pq.read_table(ref_path)
                validate_reference_table(table)
                self.assertIn("proto_ids", table.column_names)
                self.assertIn("proto_weights", table.column_names)
                self.assertIn("mean_distance", table.column_names)
                self.assertIn("null_mean", table.column_names)
                self.assertIn("bio_dispersion", table.column_names)
                self.assertNotIn("nearest_ref_indices", table.column_names)
                self.assertNotIn("null_std", table.column_names)
                self.assertEqual(len(table.column("null_mean")), 4)
                self.assertEqual(len(table.column("bio_dispersion")), 4)


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
                        "extract",
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

            samples = [read_sample_buffers(path) for path in sample_paths]
            self.assertEqual(samples[0].num_rows, samples[1].num_rows)
            self.assertEqual(samples[0].meta.contig_order, ("chr21",))

            sheet_path = root / "samplesheet.yaml"
            sheet_path.write_text(
                yaml.safe_dump([{"filepath": str(p)} for p in sample_paths]),
                encoding="utf-8",
            )
            reference_prefix = root / "reference"
            result = runner.invoke(
                app,
                [
                    "reference",
                    "--samplesheet",
                    str(sheet_path),
                    "--prefix",
                    str(reference_prefix),
                    "--bin-size",
                    str(PUBLISHED_BIN_SIZE),
                    "--min-cohort-size",
                    "2",
                    "--pca-components",
                    "1",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)

            table = pq.read_table(parquet_path(reference_prefix))
            validate_reference_table(table)
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
            contigs_dict = read_fasta_contigs(fasta)
            cohort = cohort_from_fasta(contigs_dict, 100, fasta_path=fasta)

            self.assertEqual(cohort.bin_size, 100)
            self.assertEqual(
                cohort.contigs, ("chr1", "chr1", "chr1", "chr2", "chr2", "chrX")
            )
            np.testing.assert_array_equal(cohort.starts, [0, 100, 200, 0, 100, 0])
            np.testing.assert_array_equal(cohort.ends, [100, 200, 250, 100, 150, 100])
            self.assertEqual(cohort.counts.shape, (6, 0))

            prefix = Path(tmpdir) / "ref"
            run(
                ReferenceConfig(
                    prefix=prefix,
                    fasta=fasta,
                    bin_size=100,
                    pca_components=1,
                    min_cohort_size=2,
                )
            )
            table = pq.read_table(parquet_path(prefix))
            validate_reference_table(table)
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
            self.assertTrue((Path(tmpdir) / "ref_qc.json").is_file())

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
                    prefix=reference_prefix,
                    fasta=fasta,
                    bin_size=100,
                    pca_components=1,
                    min_cohort_size=2,
                    qc_report=False,
                )
            )
            self.assertFalse((root / "ref_qc.json").is_file())
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
                    "--fasta",
                    str(fasta),
                    "--prefix",
                    str(prefix),
                    "--bin-size",
                    "100",
                    "--min-cohort-size",
                    "2",
                    "--pca-components",
                    "1",
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
                        entries=(SamplesheetEntry(filepath=Path("sample.ccnv")),),
                        prefix=Path(tmpdir) / "ref",
                        fasta=fasta,
                        bin_size=100,
                    )
                )

    def test_neither_samples_nor_fasta_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ReferenceError, "sample Parquet entries or a FASTA"
        ):
            run(ReferenceConfig(prefix=Path("ref")))

    def test_prenatal_mode_rejects_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(Path(tmpdir) / "genome.fa", (("chr1", 100),))
            with self.assertRaisesRegex(
                ReferenceError,
                "Flat FASTA references are only supported in --mode germline; "
                "--mode prenatal requires a cohort --samplesheet.",
            ):
                run(
                    ReferenceConfig(
                        prefix=Path(tmpdir) / "ref",
                        fasta=fasta,
                        bin_size=100,
                        mode="prenatal",
                    )
                )


def _write_tiny_fasta(path: Path, records: tuple[tuple[str, int], ...]) -> Path:
    with path.open("w", encoding="ascii") as handle:
        for name, length in records:
            handle.write(f">{name}\n{'A' * length}\n")
    pysam.faidx(str(path))
    return path


if __name__ == "__main__":
    unittest.main()
