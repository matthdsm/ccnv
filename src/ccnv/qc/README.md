# `ccnv.qc`

## Overview

The `ccnv.qc` package implements quality control accounting, reference model auditing, usability verdict evaluation (`PASS`, `WARN`, `FAIL`), and structured JSON report persistence. It tracks read filtering drop counters on binned target contigs, computes neighbor distance distributions and usable bin percentages for reference branches, and compiles single-sample prediction summaries including prenatal metrics and discordance flags.

## Interface

- `[src/ccnv/qc/sample.py:QC_COUNTERS]`: Tuple of standardized read filtering counter identifiers.
- `[src/ccnv/qc/sample.py:empty_qc]`: Initializes a zeroed counter dictionary for worker process streaming.
- `[src/ccnv/qc/sample.py:build_sample_qc]`: Compiles and verifies read count totals and drop reasons on target contigs.
- `[src/ccnv/qc/sample.py:sanitize_qc_payload]`: Recursively converts NumPy scalars and non-standard types to JSON-serializable primitives.
- `[src/ccnv/qc/reference.py:compute_branch_qc]`: Computes usable bin counts, masked fractions, and neighbor Euclidean distance statistics for a reference branch.
- `[src/ccnv/qc/reference.py:evaluate_autosomal_or_female_verdict]`: Determines the reference usability verdict (`PASS`, `WARN`, `FAIL`) for autosomal and female reference models.
- `[src/ccnv/qc/reference.py:evaluate_male_verdict]`: Determines the reference usability verdict for male sex reference branches.
- `[src/ccnv/qc/reference.py:run_reference_qc_on_table]`: Executes full reference auditing across all active branches.
- `[src/ccnv/qc/reference.py:write_reference_qc_json]`: Writes `<prefix>_qc.json` reference audit summaries to disk.
- `[src/ccnv/qc/predict.py:build_predict_qc_payload]`: Compiles sample prediction statistics, segmentation parameters, and prenatal findings into a unified payload.
- `[src/ccnv/qc/predict.py:write_predict_qc_json]`: Writes `<prefix>_qc.json` sample prediction quality summaries to disk.

## Invariants

- `[src/ccnv/qc/sample.py:build_sample_qc]`: Read counters must be non-negative integers satisfying balance identity $\text{examined} = \text{kept} + \sum \text{filtered}$.
- `[src/ccnv/qc/reference.py:evaluate_autosomal_or_female_verdict]`: Severity ranks $\text{PASS} \lt \text{WARN} \lt \text{FAIL}$; triggers failure when usable bin fraction falls below 60% or mean neighbor distance exceeds the 3-sigma outlier threshold.
- `[src/ccnv/qc/predict.py:build_predict_qc_payload]`: Predict QC payload requires valid execution configuration, contig statistics, and well-typed prenatal blocks when analysis mode is prenatal.

## Side Effects

- `[src/ccnv/qc/reference.py:write_reference_qc_json]`: Serializes `<prefix>_qc.json` to disk.
- `[src/ccnv/qc/predict.py:write_predict_qc_json]`: Serializes `<prefix>_qc.json` to disk.

## Verification

- Run: `pytest`
