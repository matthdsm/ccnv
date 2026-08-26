# Benchmarks

Benchmarks are descriptive and intentionally have no CI pass/fail thresholds.
Reproduce the recorded run with:

```console
pixi run python benchmarks/run_benchmarks.py \
  --seed 20260824 --scale 2 --repeats 3 \
  --output benchmark-results.json
```

## Recorded run

Recorded on macOS 26.6.2 arm64, Python 3.12.14, and NumPy 2.5.2. Inputs were
4,000 alignment records, 1,000 bins by 20 samples for neighbor construction,
1,000 prediction bins, and 4,000 segmentation bins. Values below are median
wall-clock times across three runs:

- BAM conversion and sample Parquet write: 0.0072 seconds.
- Batched off-contig neighbor construction: 0.0275 seconds.
- Scikit-learn brute-distance comparison: 0.0050 seconds. This comparison does
  less post-processing than the production path, but demonstrates that the
  current row-wise off-contig selection remains an optimization opportunity.
- Prediction normalization: 0.0295 seconds.
- Selected weighted Binseg: 0.0120 seconds.
- Weighted PELT on the identical signal: 5.52 seconds. Binseg was about 459
  times faster on this input.
- Whole-process peak RSS: 212,779,008 bytes (203 MiB).
- Zstd sample Parquet artifact: 16,785 bytes.

The first conversion repetition includes more setup overhead (0.0294 seconds);
the minimum/median/maximum values and complete machine metadata are emitted in
the JSON result.

## Interpretation and limits

The benchmark script fixes the random seed, records input sizes and machine
metadata, and runs all implementations on the same in-memory data. It includes
the nearest-neighbor library baseline and the rejected PELT segmentation
candidate.

The upstream 1.3.2 executable is not part of the Python environment and its
DNAcopy stage requires R, which this package deliberately does not install.
Consequently, direct upstream timings for conversion, normalization, and
DNAcopy are not reproducible in the R-free wheel smoke environment. The
immutable upstream fixtures support correctness comparisons but contain no
timing or peak-RSS provenance. This is the remaining performance-validation
gap; results must not be presented as an end-to-end speedup over 1.3.2.

Peak RSS is process-wide rather than stage-isolated, and timings are
microbenchmarks rather than clinical-cohort capacity guarantees.
