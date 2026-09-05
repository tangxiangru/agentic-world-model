# Official evaluator evidence at harvest

The operator now preserves the regular evaluator evidence files under every
`official_eval/` attempt and every `.official-inspect-*` snapshot. This includes
failed attempts and raw Inspect JSON larger than the normal 2 MiB text cap.
Weights, public runtime binaries and unsupported files are not silently copied
as logs: unexpected files make retention partial and are listed as errors.

`official-evidence/index.json` maps original result-relative paths to exact-byte
gzip archives, recording compressed and uncompressed fingerprints. It binds the
copied metrics and runtime provenance. `status.json` records the index fingerprint
separately, plus `preserved`, `partial`, `failed` or `absent` retention state.

```python
from awm.ptb_evidence_retention import verify_official_evidence

# Obtain this fingerprint from the independently retained operator status,
# e.g. the committed harvest record, never from an untrusted index itself.
index = verify_official_evidence(
    bundle,
    expected_index_fingerprint=status["official_evidence_retention"]["index_fingerprint"],
)
```

Verification reads and parses the same checked metadata bytes, verifies archive
bytes and their decompressed content, and rechecks the raw-log binding in metrics.
It does not read the original result volume. Source and output traversal reject
symlinks; publication cannot overwrite or delete a preexisting archive. Source
inventory/read failures or observed modifications prevent full-retention claims.
Direct capture calls require a fresh destination; retries use the existing
operator harvest lifecycle rather than mutating an existing archive in place.

Retention is not scientific validation. Scientific completion, judge flags,
placement quarantine, and failed-job facts still come from the frozen task
validator and independent receipt identity. A retention I/O failure is recorded
without hiding those facts or stopping the remaining harvest. The archives are
evidence, not a runnable final-model bundle; no model weights are retained here.
The external index fingerprint is the integrity anchor. If an adversary can
rewrite both the anchor and the archives, checksums alone cannot authenticate
the result.

Validation: 76 relevant tests passed, including original-volume removal,
large logs, failed attempts, altered/deleted archives and indices, missing raw
bindings, traversal failures, source changes, publication conflicts and output
symlinks, retention failures, and legacy completion/judge/placement behavior.
Independent review is recorded in the planner's
`analysis-2026-09-05-opus48-wave1/retention-review.md`.

This engineering checkpoint does not admit HumanEval, submit a job or establish
compute-node acceptance. Node/GPU/outer-timeout acceptance and task admission
remain separate prerequisites.
