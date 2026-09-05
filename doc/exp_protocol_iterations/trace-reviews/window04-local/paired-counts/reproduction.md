# Planner reproduction and retained evidence

The planner read the full focused report and233-line original script, then reran the structural parser on all34 receipt-backed developer logs. Output reproduced the committed `inventory.tsv` and `pairs.tsv` **byte-for-byte**:

- inventory SHA256 `fbcd2f1b0595ff34d7c0dccf70b059dd00fb01e19c81d7b63a29cbf76dbacd57`
- pairs SHA256 `7c9b92eeeb6524a6e9387e11a55ffaff57afeeda058817295fb0b57c0b27731d`

Independent post-run assertions checked every source's empty issue list, pair-table denominator sums and zero aligned input/target/prompt mismatches. The saved script now includes those assertions and requires an explicit existing empty `--output-dir`; only output routing/validation changed from the reviewer's original hash recorded in its report. It does not execute models/evaluators or change Slurm.

Reproduce from the repository with Python standard library, pointing `--output-dir` at a fresh private temporary directory:

```bash
.venv/bin/python doc/exp_protocol_iterations/trace-reviews/window04-local/paired-counts/audit_pairs.py --output-dir /absolute/path/to/empty-temporary-directory
```

`structural-audit-summary.json` retains source hashes, receipt/status lineage, full logged protocol metadata, sample/count checks and nine exact pair tables. It intentionally omits the9MB full artifact's per-sample rows, declared ID arrays, discordant-ID arrays and set-difference arrays. The unabridged source output was `/tmp/window04-control-pairs.K96WXQ/structural-audit.json`, SHA256 `8ec9db6bc17e33e69d49c9001c81b363ce2c6c719a2d8371e56aa4a97856fe41`; the replay at `/tmp/window04-control-pairs-replay-RFqGESYz/structural-audit.json` regenerates those arrays. These temporary paths are not claimed as permanent archival storage. The committed reproducible parser and exact original-source inventory reconstruct the omitted fields. No question text or raw benchmark item body is copied into this compact audit.

Important interpretation: no sample is missing in the paired comparisons; the prior regex tables were materially wrong. Corrected two-sided tests do not prove equivalence, and c01s07's changed-token-cap/changed-n contrast does not estimate isolated repeat noise. Preserve the original helper reports unedited and attach this correction to synthesis before final candidate adjudication.
