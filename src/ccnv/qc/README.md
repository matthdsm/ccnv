# `ccnv.qc`

## Overview

The `ccnv.qc` package implements sample read accounting, reference model usability auditing (`PASS`, `WARN`, `FAIL`), and structured JSON payload assembly (`<prefix>_qc.json`). It records deterministic read filtering counters on target contigs, evaluates reference branch neighbor distance dispersion and usable `chrY` fractions (plus prenatal `prenatal_autosomal`/`prenatal_ffy` model summaries), and compiles single-sample prediction payloads covering read metrics, CNV call counts, fragmentomic pre-analytical metrics, and prenatal findings.

## Interface

- [`sample.py:QC_COUNTERS`](./sample.py), [`empty_qc`](./sample.py), [`build_sample_qc`](./sample.py), & [`sanitize_qc_payload`](./sample.py): Deterministic 16-counter read/filter map builders (`header_mapped`, `reads_seen`, `reads_kept`, `effective_reads_examined`, `effective_reads_kept`, `autosomal_reads_kept`, `sex_reads_kept`, and `filter_*` drop reasons) and recursive NumPy/float JSON sanitizers.
- [`reference.py:DEFAULT_MIN_REF_BINS`](./reference.py) (`150`), [`OUTLIER_N_SIGMA`](./reference.py) (`3.0`), [`CHRY_MIN_USABLE_PCT_WARN`](./reference.py) (`50.0`), [`CHRY_MIN_USABLE_PCT_FAIL`](./reference.py) (`20.0`), [`CHRY_MEAN_DIST_WARN`](./reference.py) (`5.0`), & [`CHRY_MEAN_DIST_FAIL`](./reference.py) (`100.0`): Reference quality threshold constants alongside `VERDICT_SEVERITY` and `SEVERITY_TO_VERDICT`.
- [`reference.py:compute_branch_qc`](./reference.py) & [`compute_signal_contig_metrics`](./reference.py): Compute unmasked usable bin counts, dimension-normalized prototype distances, $`3\sigma`$ outlier bin percentages, low-prototype rates (`p_low`), and sex-signal contig (`chrY`) metrics per reference branch.
- [`reference.py:evaluate_autosomal_or_female_verdict`](./reference.py) & [`evaluate_male_verdict`](./reference.py): Escalate branch usability verdicts across `PASS < WARN < FAIL` based on `std_of_means` (`autosomal`/`female_sex`), `mean_of_means` and `chrY` coverage (`male_sex`, including `0%` usable `chrY -> FAIL`), $`3\sigma`$ outlier rates (`> 1.0%`), and low-prototype rates (`p_low > 0.005`).
- [`reference.py:run_reference_qc_on_table`](./reference.py) & [`write_reference_qc_json`](./reference.py): Audit an in-memory reference Arrow table (including `prenatal` metadata/branch summaries and flat-FASTA `WARN` escalation), optionally write `<prefix>_qc.json`, and emit human-readable verdict summaries to `stderr`.
- [`predict.py:build_predict_qc_payload`](./predict.py) & [`write_predict_qc_json`](./predict.py): Construct and persist prediction QC JSON (`schema_version`, `mode`, `sample`, `read_metrics` with `lambda_gc`, `cnv_summary`, `fragmentomics` with `"WARN_HIGH_LEUKOCYTE_DNA"` when `high_mw_fraction > 0.05`, and `prenatal` including `active_ff_se`, `origin_summary`, `lod_ff_35mb`, and `lod_ff_3mb`).

## Invariants

- [`sample.py:sanitize_qc_payload`](./sample.py): Replaces non-finite floats (`NaN`, `Inf`) with `None` and rounds finite floats to 6 decimal places for deterministic JSON output.
- [`reference.py:run_reference_qc_on_table`](./reference.py): Enforces monotonic severity escalation (`PASS (0) < WARN (1) < FAIL (2)`) across audited branches and caps flat FASTA references (`source == "fasta"`) at `WARN`.
- [`predict.py:build_predict_qc_payload`](./predict.py): Populates `prenatal` strictly when `mode == "prenatal"` and `prenatal_result` is provided, and `fragmentomics` whenever sample metadata contains `ccnv.fragmentomics`.

## Side Effects

- [`reference.py:run_reference_qc_on_table`](./reference.py) & [`write_reference_qc_json`](./reference.py): Write `<prefix>_qc.json` when `json_path` is supplied and print formatted reference QC verdicts to `sys.stderr` when `emit_stderr=True`.
- [`predict.py:write_predict_qc_json`](./predict.py): Creates parent directories and writes `<prefix>_qc.json` to disk.

## Verification

- Run: `pixi run test`
