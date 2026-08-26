import csv
import math
import unittest
from pathlib import Path

import numpy as np

from ccnv.predict import call_segment
from ccnv.segment import segment_contig

FIXTURES = Path(__file__).parent / "fixtures" / "upstream"
BIN_SIZE = 100_000


class TestUpstreamNumericAgreement(unittest.TestCase):
    def test_synthetic_segment_means_and_calls_are_concordant(self) -> None:
        bins_by_contig: dict[str, list[tuple[float, float]]] = {}
        with (FIXTURES / "synthetic" / "normalized_bins.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                bins_by_contig.setdefault(row["chr"], []).append(
                    (float(row["ratio"]), float(row["weight"]))
                )
        with (FIXTURES / "synthetic" / "dnacopy_segments.tsv").open() as handle:
            expected_segments = {
                row["chr"]: (
                    int(row["start_bin"]),
                    int(row["end_bin"]),
                    float(row["ratio"]),
                )
                for row in csv.DictReader(handle, delimiter="\t")
            }
        with (FIXTURES / "synthetic" / "calls.tsv").open() as handle:
            expected_calls = {
                (row["chr"], row["type"])
                for row in csv.DictReader(handle, delimiter="\t")
            }

        observed_calls: set[tuple[str, str]] = set()
        absolute_errors: list[float] = []
        for contig, rows in bins_by_contig.items():
            ratios, weights = zip(*rows, strict=True)
            segments = segment_contig(
                contig=contig,
                starts=np.arange(len(rows)) * BIN_SIZE,
                ends=np.arange(1, len(rows) + 1) * BIN_SIZE,
                ratios=ratios,
                weights=weights,
                contig_length=len(rows) * BIN_SIZE,
                penalty=0.01,
            )
            self.assertEqual(len(segments), 1)
            expected_start, expected_end, expected_ratio = expected_segments[contig]
            self.assertLessEqual(
                abs(segments[0].start - expected_start * BIN_SIZE),
                BIN_SIZE,
            )
            self.assertLessEqual(
                abs(segments[0].end - expected_end * BIN_SIZE),
                BIN_SIZE,
            )
            absolute_errors.append(abs(segments[0].mean - expected_ratio))
            call = call_segment(
                segments[0].mean,
                math.nan,
                z_threshold=3.0,
                ratio_threshold=0.2,
            )
            if call is not None:
                observed_calls.add((contig, call))

        self.assertLessEqual(float(np.max(absolute_errors)), 1e-12)
        self.assertEqual(observed_calls, expected_calls)

    def test_real_fixture_gain_is_concordant_after_coordinate_conversion(self) -> None:
        bins: dict[str, list[tuple[int, int, float, float]]] = {}
        with (FIXTURES / "real" / "ID_bins.bed").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                ratio = float(row["ratio"])
                bins.setdefault(row["chr"], []).append(
                    (
                        int(row["start"]) - 1,
                        int(row["end"]),
                        ratio,
                        float(math.isfinite(ratio)),
                    )
                )
        with (FIXTURES / "real" / "ID_aberrations.bed").open() as handle:
            expected = [
                (
                    row["chr"],
                    int(row["start"]) - 1,
                    int(row["end"]),
                    row["type"],
                )
                for row in csv.DictReader(handle, delimiter="\t")
            ]

        observed: list[tuple[str, int, int, str]] = []
        for contig, rows in bins.items():
            columns = tuple(zip(*rows, strict=True))
            for segment in segment_contig(
                contig=contig,
                starts=columns[0],
                ends=columns[1],
                ratios=columns[2],
                weights=columns[3],
                contig_length=int(columns[1][-1]),
                penalty=0.05,
            ):
                call = call_segment(
                    segment.mean,
                    math.nan,
                    z_threshold=3.0,
                    ratio_threshold=0.05,
                )
                if call is not None:
                    observed.append((contig, segment.start, segment.end, call))

        self.assertEqual(len(expected), 1)
        matches = [
            item
            for item in observed
            if item[0] == expected[0][0]
            and item[3] == expected[0][3]
            and abs(item[1] - expected[0][1]) <= BIN_SIZE
            and abs(item[2] - expected[0][2]) <= BIN_SIZE
        ]
        recall = len(matches) / len(expected)
        precision = len(matches) / len(observed)
        self.assertEqual(recall, 1.0)
        self.assertGreaterEqual(precision, 0.5)


if __name__ == "__main__":
    unittest.main()
