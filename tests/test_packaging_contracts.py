"""Contract tests for multi-arch Dockerfile and GitHub Actions packaging workflows."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestPackagingContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_dockerfile_copies_rust_crates_and_strips_build_artifacts(self) -> None:
        dockerfile = (self.repo_root / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("Cargo.toml", dockerfile)
        self.assertIn("Cargo.lock", dockerfile)
        self.assertIn("crates", dockerfile)
        self.assertIn("FROM ghcr.io/prefix-dev/pixi:0.84.1 AS build", dockerfile)
        self.assertIn("FROM ubuntu:26.04 AS production", dockerfile)
        self.assertIn("rm -rf", dockerfile)
        self.assertIn("rustlib", dockerfile)

    def test_ci_preserves_action_versions_and_expands_multiarch_matrix(self) -> None:
        ci = (self.repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        # Critical action version pins
        self.assertIn("actions/checkout@v7", ci)
        self.assertIn("docker/build-push-action@v7.4.0", ci)
        self.assertIn("9936258fee697aff89b3fed6ca1f80a382ef694e", ci)
        self.assertIn("docker/setup-qemu-action@v4", ci)
        self.assertIn("docker/setup-buildx-action@v4", ci)
        self.assertIn("docker/login-action@v4", ci)

        # Multi-arch platform coverage and maturin commands
        self.assertIn("aarch64", ci)
        self.assertIn("macos", ci)
        self.assertIn("ubuntu-24.04-arm", ci)
        self.assertIn("maturin build --release", ci)
        self.assertIn("maturin sdist", ci)
        self.assertIn("pixi run docs-build", ci)
