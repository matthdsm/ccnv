"""Reproducible, threshold-free microbenchmarks for pipeline hot paths."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pysam
from sklearn.neighbors import NearestNeighbors

from ccnv.convert import run as convert
from ccnv.models import ConvertConfig, PredictConfig, parquet_path
from ccnv.predict import BranchModel, normalize_branch
from ccnv.reference import BRANCH_AUTOSOMAL, select_nearest_bins
from ccnv.segment import segment_contig


def benchmark(seed: int, scale: int, repeats: int) -> dict[str, object]:
    """Run all stages on deterministic synthetic inputs."""

    rng = np.random.default_rng(seed)
    timings: dict[str, dict[str, float]] = {}
    artifact_sizes: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bam = root / "synthetic.bam"
        sample = root / "synthetic"
        _write_bam(bam, scale * 2_000)
        timings["convert"] = _measure(
            lambda: convert(ConvertConfig(bam, sample, bin_size=1_000)),
            repeats,
        )
        artifact_sizes["sample_parquet_bytes"] = parquet_path(sample).stat().st_size

        profiles = rng.normal(size=(scale * 500, 20))
        contig_ids = np.arange(scale * 500) % 5
        timings["reference_neighbors"] = _measure(
            lambda: select_nearest_bins(
                profiles,
                contig_ids,
                k=20,
                max_batch_memory_mb=64,
            ),
            repeats,
        )
        timings["reference_neighbors_sklearn_brute"] = _measure(
            lambda: (
                NearestNeighbors(
                    n_neighbors=21,
                    algorithm="brute",
                    metric="euclidean",
                )
                .fit(profiles)
                .kneighbors(profiles)
            ),
            repeats,
        )

        model = _prediction_model(rng, scale * 500)
        config = PredictConfig(
            Path("sample"),
            Path("reference"),
            Path("output"),
            min_reference_bins=3,
        )
        timings["predict_normalization"] = _measure(
            lambda: normalize_branch(model, config),
            repeats,
        )

        n_segment_bins = scale * 2_000
        ratios = np.concatenate(
            (
                rng.normal(0.0, 0.03, n_segment_bins // 2),
                rng.normal(0.35, 0.03, n_segment_bins - n_segment_bins // 2),
            )
        )
        timings["segmentation"] = _measure(
            lambda: segment_contig(
                contig="synthetic",
                starts=np.arange(n_segment_bins) * 1_000,
                ends=np.arange(1, n_segment_bins + 1) * 1_000,
                ratios=ratios,
                weights=np.ones(n_segment_bins),
                contig_length=n_segment_bins * 1_000,
                penalty=1.0,
            ),
            repeats,
        )
        timings["segmentation_pelt"] = _measure(
            lambda: segment_contig(
                contig="synthetic",
                starts=np.arange(n_segment_bins) * 1_000,
                ends=np.arange(1, n_segment_bins + 1) * 1_000,
                ratios=ratios,
                weights=np.ones(n_segment_bins),
                contig_length=n_segment_bins * 1_000,
                penalty=1.0,
                algorithm="pelt",
            ),
            repeats,
        )

    return {
        "benchmark_version": 1,
        "seed": seed,
        "scale": scale,
        "repeats": repeats,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "inputs": {
            "convert_reads": scale * 2_000,
            "reference_bins": scale * 500,
            "reference_samples": 20,
            "predict_bins": scale * 500,
            "segmentation_bins": scale * 2_000,
        },
        "timings_seconds": timings,
        "peak_rss_bytes": _peak_rss_bytes(),
        "artifact_sizes": artifact_sizes,
    }


def _measure(function: object, repeats: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        durations.append(perf_counter() - started)
    return {
        "minimum": min(durations),
        "median": float(np.median(durations)),
        "maximum": max(durations),
    }


def _prediction_model(rng: np.random.Generator, n_bins: int) -> BranchModel:
    contig_ids = np.arange(n_bins) % 5
    nearest = tuple(
        np.flatnonzero(contig_ids != contig_ids[index])[:20].astype(np.int64)
        for index in range(n_bins)
    )
    return BranchModel(
        branch=BRANCH_AUTOSOMAL,
        contigs=tuple(f"contig-{value}" for value in contig_ids),
        starts=np.arange(n_bins, dtype=np.int64) * 1_000,
        ends=np.arange(1, n_bins + 1, dtype=np.int64) * 1_000,
        counts=rng.integers(10, 100, size=n_bins).astype(np.float64),
        masked=np.zeros(n_bins, dtype=np.bool_),
        pca_mean=np.full(n_bins, 1.0 / n_bins),
        pca_loadings=np.zeros((2, n_bins)),
        nearest_indices=nearest,
        nearest_distances=tuple(np.linspace(0.01, 0.2, 20) for _ in range(n_bins)),
        null_ratios=tuple(rng.normal(0.0, 0.05, 20) for _ in range(n_bins)),
    )


def _write_bam(path: Path, n_reads: int) -> None:
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "synthetic", "LN": 2_000_000}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index in range(n_reads):
            read = pysam.AlignedSegment()
            read.query_name = f"r{index}"
            read.query_sequence = "A" * 10
            read.flag = 0
            read.reference_id = 0
            read.reference_start = index * 997 % 1_999_990
            read.mapping_quality = 60
            read.cigar = ((0, 10),)
            read.query_qualities = pysam.qualitystring_to_array("I" * 10)
            bam.write(read)
    pysam.sort("-o", str(path.with_suffix(".sorted.bam")), str(path))
    path.with_suffix(".sorted.bam").replace(path)
    pysam.index(str(path))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1_024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.scale < 1 or arguments.repeats < 1:
        parser.error("--scale and --repeats must be positive")
    result = benchmark(arguments.seed, arguments.scale, arguments.repeats)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
