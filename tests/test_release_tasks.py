"""Unit and integration tests for Pixi bump-version and bump-schema maintenance tasks."""

from __future__ import annotations

import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

from ccnv.models.metadata import SCHEMA_VERSION


def _load_bump_module(repo_root: Path) -> ModuleType:
    script_path = repo_root / "scripts" / "bump.py"
    spec = importlib.util.spec_from_file_location("ccnv_bump_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot_repo_text_tree(repo_root: Path, dest_root: Path) -> None:
    rel_files = (
        "pyproject.toml",
        "Cargo.toml",
        "Cargo.lock",
        "zensical.toml",
        "README.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "LICENSE.md",
    )
    for rel in rel_files:
        src = repo_root / rel
        if src.is_file():
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for folder in ("src", "docs", "tests"):
        base = repo_root / folder
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            rel_path = path.relative_to(repo_root)
            if rel_path.name == "test_release_tasks.py" or any(
                part in {"__pycache__", "superpowers", "designs"}
                for part in rel_path.parts
            ):
                continue
            dst = dest_root / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def _ensure_unreleased_entry(changelog_path: Path, note_line: str) -> None:
    text = changelog_path.read_text(encoding="utf-8")
    if "## [Unreleased]" not in text:
        text = text.rstrip() + "\n\n## [Unreleased]\n"
    updated = re.sub(
        r"(?m)^## \[Unreleased\]\s*$",
        f"## [Unreleased]\n\n### Changed\n\n- {note_line}",
        text,
        count=1,
    )
    changelog_path.write_text(updated, encoding="utf-8")


class TestReleaseAndSchemaPixiTasks(unittest.TestCase):
    repo_root: Path
    bump_mod: ModuleType

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.bump_mod = _load_bump_module(cls.repo_root)

    def test_pixi_tasks_registered_and_dry_run_resolves(self) -> None:
        pyproject = tomllib.loads(
            (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        tasks = pyproject["tool"]["pixi"]["tasks"]
        self.assertEqual(tasks["bump-version"], "python scripts/bump.py version")
        self.assertEqual(tasks["bump-schema"], "python scripts/bump.py schema")

        pixi_bin = shutil.which("pixi")
        if pixi_bin is not None:
            for cmd, expected in (
                (
                    [pixi_bin, "run", "--no-config", "-n", "bump-version", "0.2.0"],
                    "python scripts/bump.py version 0.2.0",
                ),
                (
                    [pixi_bin, "run", "--no-config", "-n", "bump-schema", "3.0.0"],
                    "python scripts/bump.py schema 3.0.0",
                ),
            ):
                proc = subprocess.run(
                    cmd, cwd=self.repo_root, capture_output=True, text=True, check=False
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn(expected, proc.stderr + proc.stdout)

    def test_bump_version_lifecycle_across_pre1_dev_and_stable_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _snapshot_repo_text_tree(self.repo_root, root)
            _ensure_unreleased_entry(root / "CHANGELOG.md", "Prepared 0.91.0 release.")

            out = io.StringIO()
            with redirect_stdout(out):
                rc = self.bump_mod.main(
                    ["--root", str(root), "version", "v0.91.0", "--date", "2026-11-18"]
                )
            self.assertEqual(rc, 0)
            self.assertIn("Bumped application version to 0.91.0", out.getvalue())

            pyproject = tomllib.loads(
                (root / "pyproject.toml").read_text(encoding="utf-8")
            )
            cargo_toml = tomllib.loads(
                (root / "Cargo.toml").read_text(encoding="utf-8")
            )
            cargo_lock = tomllib.loads(
                (root / "Cargo.lock").read_text(encoding="utf-8")
            )
            self.assertEqual(pyproject["project"]["version"], "0.91.0")
            self.assertEqual(cargo_toml["package"]["version"], "0.91.0")
            ccnv_core_pkgs = [
                p for p in cargo_lock["package"] if p["name"] == "ccnv-core"
            ]
            self.assertEqual(len(ccnv_core_pkgs), 1)
            self.assertEqual(ccnv_core_pkgs[0]["version"], "0.91.0")

            init_py = (root / "src" / "ccnv" / "__init__.py").read_text(
                encoding="utf-8"
            )
            self.assertIn('__version__ = "0.91.0"', init_py)

            cl_091 = (root / "CHANGELOG.md").read_text(encoding="utf-8")
            self.assertIn("## [Unreleased]\n\n## [0.91.0] - 2026-11-18\n", cl_091)
            self.assertIn(
                "[Unreleased]: https://github.com/matthdsm/ccnv/compare/v0.91.0...HEAD",
                cl_091,
            )

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                dup_rc = self.bump_mod.main(
                    ["--root", str(root), "version", "0.91.0", "--date", "2026-11-18"]
                )
            self.assertEqual(dup_rc, 1)
            self.assertIn("already contains release section", err_buf.getvalue())

            empty_err = io.StringIO()
            with redirect_stderr(empty_err):
                empty_rc = self.bump_mod.main(
                    ["--root", str(root), "version", "0.91.1", "--date", "2026-11-19"]
                )
            self.assertEqual(empty_rc, 1)
            self.assertIn("no entries to finalize", empty_err.getvalue())

            ver_dev, _ = self.bump_mod.bump_version(root, "0.92.0dev")
            self.assertEqual(ver_dev, "0.92.0dev")
            cargo_dev = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertEqual(cargo_dev["package"]["version"], "0.92.0-dev")
            self.assertNotIn(
                "## [0.92.0dev]", (root / "CHANGELOG.md").read_text(encoding="utf-8")
            )

            _ensure_unreleased_entry(root / "CHANGELOG.md", "Stabilized 99.0.0 API.")
            ver_stable, _ = self.bump_mod.bump_version(
                root, "99.0.0", release_date="2026-12-01"
            )
            self.assertEqual(ver_stable, "99.0.0")
            for doc_rel in ("README.md", "CHANGELOG.md", "docs/index.md"):
                txt = (root / doc_rel).read_text(encoding="utf-8")
                self.assertNotIn("active pre-1.0 alpha development", txt)
                self.assertNotIn("\n\n\n", txt)

            cl_stable = (root / "CHANGELOG.md").read_text(encoding="utf-8")
            self.assertIn("## [99.0.0] - 2026-12-01\n\n### Changed", cl_stable)
            self.assertIn(
                "[Unreleased]: https://github.com/matthdsm/ccnv/compare/v99.0.0...HEAD",
                cl_stable,
            )
            self.assertIn(
                "[99.0.0]: https://github.com/matthdsm/ccnv/compare/v0.91.0...v99.0.0",
                cl_stable,
            )

    def test_bump_schema_updates_models_docs_and_tests_preserving_semver_url_and_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _snapshot_repo_text_tree(self.repo_root, root)
            orig_schema = SCHEMA_VERSION
            s_maj, s_min, _ = (int(x) for x in orig_schema.split("."))

            _ensure_unreleased_entry(
                root / "CHANGELOG.md",
                f"Historical release using schema (`{orig_schema}`).",
            )
            self.bump_mod.bump_version(root, "0.91.0", release_date="2026-11-18")
            _ensure_unreleased_entry(
                root / "CHANGELOG.md",
                f"Evolved `.ccnv` schema (`{orig_schema}`).",
            )

            expected_minor = f"{s_maj}.{s_min + 1}.0"
            new_v1, _ = self.bump_mod.bump_schema(root, "minor")
            self.assertEqual(new_v1, expected_minor)
            meta_minor = (root / "src" / "ccnv" / "models" / "metadata.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                f'SCHEMA_VERSION: Final[str] = "{expected_minor}"', meta_minor
            )
            self.assertIn(
                f'SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({{"{expected_minor}"}})',
                meta_minor,
            )

            self.bump_mod.bump_schema(root, "97.1.0")
            expected_patch = "97.1.1"
            new_v2, _ = self.bump_mod.bump_schema(root, "patch", keep_previous=True)
            self.assertEqual(new_v2, expected_patch)
            meta_patch = (root / "src" / "ccnv" / "models" / "metadata.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                'SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({"97.1.0", "97.1.1"})',
                meta_patch,
            )
            self.assertIn(
                'SUPPORTED_SCHEMA_VERSIONS = {"97.1.0", "97.1.1"}',
                (root / "src" / "ccnv" / "models" / "README.md").read_text(
                    encoding="utf-8"
                ),
            )

            target_major = "98.0.0"
            out = io.StringIO()
            with redirect_stdout(out):
                rc = self.bump_mod.main(["--root", str(root), "schema", "major"])
            self.assertEqual(rc, 0)
            self.assertIn(
                f"Bumped Parquet schema version to {target_major}", out.getvalue()
            )

            cl_final = (root / "CHANGELOG.md").read_text(encoding="utf-8")
            self.assertIn("https://semver.org/spec/v2.0.0.html", cl_final)
            self.assertIn(f"Evolved `.ccnv` schema (`{target_major}`).", cl_final)
            self.assertIn(
                f"Historical release using schema (`{orig_schema}`).",
                cl_final,
            )

            for check_rel in (
                "AGENTS.md",
                "docs/api.md",
                "docs/cli.md",
                "docs/methodology.md",
                "docs/pipeline.md",
                "src/README.md",
                "src/ccnv/README.md",
                "src/ccnv/io/README.md",
                "src/ccnv/models/README.md",
                "src/ccnv/models/metadata.py",
                "tests/test_schemas.py",
                "tests/test_features_cohort.py",
                "tests/test_pipeline.py",
            ):
                content = (root / check_rel).read_text(encoding="utf-8")
                self.assertNotIn(
                    orig_schema, content, msg=f"stale {orig_schema} in {check_rel}"
                )
                self.assertNotIn(
                    expected_patch,
                    content,
                    msg=f"stale {expected_patch} in {check_rel}",
                )
                self.assertIn(
                    target_major, content, msg=f"missing {target_major} in {check_rel}"
                )

            err = io.StringIO()
            with redirect_stderr(err):
                same_rc = self.bump_mod.main(
                    ["--root", str(root), "schema", target_major]
                )
            self.assertEqual(same_rc, 1)
            self.assertIn(f"already {target_major}", err.getvalue())


if __name__ == "__main__":
    unittest.main()
