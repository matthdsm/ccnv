"""Contract tests for packaging, repository cleanliness, exports, and test suite layout."""

from __future__ import annotations

import ast
import importlib.resources
import io
import shutil
import subprocess
import tokenize
import tomllib
import unittest
from pathlib import Path

import ccnv._core as core_mod
import ccnv.features.fragmentomics as ff_mod
import ccnv.predict as pred_mod
from ccnv import stats
from ccnv._core import FRAG_BAND_EDGES, GC_QUANTILE_EDGES
from ccnv.io.tracks import _load_bundled_hg38_cytobands
from ccnv.models import PrenatalModelMetadata, is_canonical_contig


class TestPackagingContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_dockerfile_copies_rust_crates_and_strips_build_artifacts(self) -> None:
        dockerfile = (self.repo_root / "Dockerfile").read_text(encoding="utf-8")
        pyproject = tomllib.loads(
            (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertIn("Cargo.toml", dockerfile)
        self.assertIn("Cargo.lock", dockerfile)
        self.assertIn("crates", dockerfile)
        self.assertRegex(
            dockerfile, r"(?m)^FROM ghcr\.io/prefix-dev/pixi:\d+\.\d+\.\d+ AS build$"
        )
        self.assertIn("FROM ubuntu:26.04 AS production", dockerfile)
        self.assertIn("rm -rf", dockerfile)
        self.assertIn("rustlib", dockerfile)
        self.assertIn(
            "binutils",
            pyproject["tool"]["pixi"]["target"]["linux"]["dependencies"],
        )

    def test_ci_preserves_action_versions_and_expands_multiarch_matrix(self) -> None:
        ci = (self.repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("actions/checkout@v7", ci)
        self.assertIn("docker/build-push-action@v7.4.0", ci)
        self.assertIn("9936258fee697aff89b3fed6ca1f80a382ef694e", ci)
        self.assertIn("docker/setup-qemu-action@v4", ci)
        self.assertIn("docker/setup-buildx-action@v4", ci)
        self.assertIn("docker/login-action@v4", ci)

        self.assertIn("aarch64", ci)
        self.assertIn("macos", ci)
        self.assertIn("ubuntu-24.04-arm", ci)
        self.assertIn("maturin build --release", ci)
        self.assertIn("maturin sdist", ci)
        self.assertIn("pixi run docs-build", ci)

    def test_bundled_hg38_cytobands_resource_packaged_inside_ccnv_io(self) -> None:
        resource = importlib.resources.files("ccnv.io").joinpath("cytoBand_hg38.gz")
        self.assertTrue(resource.is_file())
        self.assertEqual(len(resource.read_bytes()), 12_118)
        self.assertFalse((self.repo_root / "cytoBand_hg38.gz").exists())
        by_contig, hg38_lengths, _ = _load_bundled_hg38_cytobands()
        self.assertEqual(len(by_contig), 711)
        self.assertEqual(sum(len(v) for v in by_contig.values()), 1_549)
        self.assertEqual(sum(1 for c in hg38_lengths if is_canonical_contig(c)), 24)
        git_bin = shutil.which("git")
        if git_bin is not None and (self.repo_root / ".git").exists():
            proc = subprocess.run(
                [git_bin, "check-ignore", "-q", "src/ccnv/io/cytoBand_hg38.gz"],
                cwd=self.repo_root,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)


class TestRepositoryCodebaseContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_repository_line_count_invariants(self) -> None:
        long_files: list[tuple[str, int]] = []

        for base_dir in ("src", "tests", "scripts"):
            for py_path in (self.repo_root / base_dir).glob("**/*.py"):
                line_count = len(py_path.read_text(encoding="utf-8").splitlines())
                if line_count >= 1_000:
                    long_files.append(
                        (str(py_path.relative_to(self.repo_root)), line_count)
                    )
        for rs_path in (self.repo_root / "crates").glob("**/*.rs"):
            line_count = len(rs_path.read_text(encoding="utf-8").splitlines())
            if line_count >= 1_000:
                long_files.append(
                    (str(rs_path.relative_to(self.repo_root)), line_count)
                )

        self.assertEqual(
            long_files,
            [],
            f"The following files exceed the 1,000-line boundary: {long_files}",
        )

    def test_ast_cleanliness_zero_any_type_ignore_bare_assert(self) -> None:
        any_violations: list[tuple[str, int]] = []
        cast_violations: list[tuple[str, int]] = []
        assert_violations: list[tuple[str, int]] = []
        type_ignore_violations: list[tuple[str, int, str]] = []

        for base_dir in ("src", "scripts"):
            for py_path in (self.repo_root / base_dir).glob("**/*.py"):
                rel_path = str(py_path.relative_to(self.repo_root))
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
                            elif alias.name == "cast":
                                cast_violations.append((rel_path, node.lineno))

        self.assertEqual(
            any_violations,
            [],
            f"Found forbidden typing.Any imports in src/: {any_violations}",
        )
        self.assertEqual(
            cast_violations,
            [],
            f"Found forbidden typing.cast imports in src/: {cast_violations}",
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

    def test_zero_pysam_or_pybigwig_imports_in_src_ccnv(self) -> None:
        forbidden_imports: list[tuple[str, int, str]] = []

        for py_path in (self.repo_root / "src" / "ccnv").glob("**/*.py"):
            rel_path = str(py_path.relative_to(self.repo_root))
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

    def test_test_suite_organization_invariants(self) -> None:
        tests_dir = self.repo_root / "tests"
        forbidden_prefixes = ("test_wave", "test_round", "test_phase", "test_defect")
        bad_filenames: list[str] = []
        top_level_tests: list[tuple[str, int, str]] = []
        any_violations: list[tuple[str, int]] = []
        cast_violations: list[tuple[str, int]] = []
        type_ignore_violations: list[tuple[str, int, str]] = []

        for py_path in sorted(tests_dir.glob("**/*.py")):
            rel_path = str(py_path.relative_to(self.repo_root))
            if py_path.name.startswith(forbidden_prefixes):
                bad_filenames.append(rel_path)

            text = py_path.read_text(encoding="utf-8")
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
                    type_ignore_violations.append(
                        (rel_path, tok.start[0], tok.string.strip())
                    )

            tree = ast.parse(text, filename=str(py_path))
            if py_path.name.startswith("test_"):
                for stmt in tree.body:
                    if isinstance(
                        stmt, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and stmt.name.startswith("test_"):
                        top_level_tests.append((rel_path, stmt.lineno, stmt.name))

            typing_aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "typing":
                            typing_aliases.add(alias.asname or "typing")
                elif isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        if alias.name == "Any":
                            any_violations.append((rel_path, node.lineno))
                        elif alias.name == "cast":
                            cast_violations.append((rel_path, node.lineno))

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in typing_aliases
                ):
                    if node.attr == "Any":
                        any_violations.append((rel_path, node.lineno))
                    elif node.attr == "cast":
                        cast_violations.append((rel_path, node.lineno))

        self.assertEqual(
            any_violations,
            [],
            f"Found forbidden typing.Any in tests/: {any_violations}",
        )
        self.assertEqual(
            cast_violations,
            [],
            f"Found forbidden typing.cast in tests/: {cast_violations}",
        )
        self.assertEqual(
            type_ignore_violations,
            [],
            f"Found forbidden # type: ignore comments in tests/: {type_ignore_violations}",
        )
        self.assertEqual(
            bad_filenames,
            [],
            f"Found forbidden temporal/defect test filenames in tests/: {bad_filenames}",
        )
        self.assertEqual(
            top_level_tests,
            [],
            f"Found top-level test functions outside unittest.TestCase: {top_level_tests}",
        )


class TestStatsPackageExports(unittest.TestCase):
    def test_all_symbols_exported(self) -> None:
        expected_symbols = [
            "CANONICAL_CHROMATOSOME_BP",
            "DEFAULT_MAX_BLACKLIST_GAP",
            "FRAG_MAX_LEN",
            "FRAG_MIN_LEN",
            "FRAG_NUM_BINS",
            "LN2_SQ",
            "LOG_WAVE_CLIP",
            "ContigSegmentationInput",
            "FragmentomicLandmarks",
            "LogitEnsembleFit",
            "MSYSexModel",
            "PoissonFFYModel",
            "SCARayClassification",
            "WeightedL2Cost",
            "WhitenedPCAResult",
            "align_histogram",
            "build_exogenous_covariate_basis",
            "build_fragmentomic_feature_vector",
            "build_whitened_subspace_features",
            "calibrate_prototype_dispersion",
            "classify_sca_ray_4d",
            "compute_autosomal_density",
            "compute_band_fisher_weights",
            "compute_fragment_length_llr",
            "compute_helical_periodicity",
            "compute_kish_loco_inflation",
            "compute_loco_prototype_support",
            "compute_motif_diversity_score",
            "deconvolve_fetal_maternal_densities",
            "encode_kmer_4",
            "estimate_dyadic_inflation",
            "estimate_sample_landmarks",
            "estimate_superblock_scales",
            "evaluate_band_weighted_counts",
            "evaluate_loco_prototype_baseline",
            "evaluate_poisson_msy_ffy",
            "exact_poisson_log2_jensen_bias",
            "exact_poisson_log_variance",
            "fit_ff_fragmentomics_ensemble",
            "fit_forward_ffx_model",
            "fit_logit_linear_ensemble",
            "fit_msy_sex_mixture",
            "fit_poisson_covariate_trends",
            "fit_poisson_msy_ffy",
            "fit_prototype_graph",
            "fit_whitened_pca",
            "fuse_active_fetal_fraction_student_t4",
            "half_sample_mode",
            "hsm_biweight_mode",
            "interpolate_dyadic_inflation",
            "mad_scale",
            "partition_contiguous_superblocks",
            "pca_irlba_numpy",
            "predict_ff_fragmentomics",
            "predict_ff_kl_deconvolution",
            "project_huber_pca_wave",
            "reconstruct_whitened_pca_expected",
            "segment_and_score_genome",
            "segment_contig",
            "squeeze_var",
            "standardize_column",
        ]
        self.assertEqual(stats.__all__, expected_symbols)
        for symbol in expected_symbols:
            self.assertTrue(hasattr(stats, symbol), f"Symbol {symbol} missing in stats")

    def test_exported_constants_match_v2_spec(self) -> None:
        expected_frag_edges = (
            30,
            50,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
            250,
            300,
            350,
            400,
            601,
        )
        expected_gc_edges = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 1.0)
        self.assertEqual(FRAG_BAND_EDGES, expected_frag_edges)
        self.assertEqual(len(FRAG_BAND_EDGES), 25)
        self.assertEqual(GC_QUANTILE_EDGES, expected_gc_edges)
        self.assertEqual(len(GC_QUANTILE_EDGES), 9)
        self.assertFalse(hasattr(core_mod, "ScanAlignmentResult"))
        self.assertFalse(hasattr(core_mod, "ScanQcCounts"))
        stub_path = Path(__file__).resolve().parents[1] / "src" / "ccnv" / "_core.pyi"
        stub_tree = ast.parse(stub_path.read_text(encoding="utf-8"))
        stub_all: list[str] = []
        for node in stub_tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                stub_all = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
        self.assertNotIn("ScanAlignmentResult", stub_all)
        self.assertNotIn("ScanQcCounts", stub_all)

    def test_legacy_align_reference_deleted_and_prenatal_metadata_contract(
        self,
    ) -> None:
        self.assertFalse(hasattr(ff_mod, "align_reference_to_weighted_sample"))
        self.assertFalse(hasattr(pred_mod, "align_reference_to_weighted_sample"))

        meta = PrenatalModelMetadata(
            prenatal_bin_size=100_000,
            yfrac_threshold=0.005,
            ffy_slope=1.0,
            olm_weights=[0.05] * 10,
            olm_bias=0.0,
            female_chrx_ratio_mean=1.0,
            female_chrx_ratio_std=0.01,
            band_fisher_weights=[1.1] * 24,
            mu_fetal_w=1.25,
            mu_maternal_w=1.0,
        )

        d = meta.to_dict()
        bw_raw = d["band_fisher_weights"]
        if not isinstance(bw_raw, list):
            self.fail("expected list for band_fisher_weights")
        self.assertEqual(len(bw_raw), 24)
        self.assertAlmostEqual(float(meta.mu_fetal_w or 0.0), 1.25)
        self.assertAlmostEqual(float(meta.mu_maternal_w or 0.0), 1.0)

        reloaded = PrenatalModelMetadata.from_dict(d)
        if reloaded.band_fisher_weights is None:
            self.fail("expected non-None band_fisher_weights")
        self.assertEqual(len(reloaded.band_fisher_weights), 24)
        self.assertAlmostEqual(float(reloaded.mu_fetal_w or 0.0), 1.25)
        self.assertAlmostEqual(float(reloaded.mu_maternal_w or 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
