# Upstream behavioral fixtures

These files are immutable behavioral observations from WisecondorX v1.3.2. They are comparison data for later scientific validation and are not accepted as inputs by the new runtime.

- `synthetic/` contains a compact normalized-bin example and its upstream DNAcopy segments, calls, and summary statistics.
- `real/` contains the published documentation example outputs: bin-level values, DNAcopy segments, and called aberrations.
- `SHA256SUMS` records fixture integrity. If a baseline must change, add a new versioned fixture instead of editing an existing observation.

The published example uses upstream's legacy 1-based inclusive BED-like coordinates. This is intentional evidence. The rewrite contract is 0-based, half-open and baseline comparison code must convert explicitly.

Unsafe legacy NPZ extraction is isolated in `tests/baseline/extract_legacy_npz.py`; never import or call it from `src/ccnv`.
