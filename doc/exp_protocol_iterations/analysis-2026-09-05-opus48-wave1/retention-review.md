# Independent official-evidence retention review — 2026-09-05

Reviewer: HumanEval-readiness subagent, informed source review. Scope: the new
retention helper, its harvest integration and corresponding synthetic tests in
`/tmp/exp-protocol-nextwave-ckg3t0gc/repo`. No training, model evaluation, queue
operation or benchmark-content inspection was performed. This review did not
rerun the tests; the main agent owns the final test results.

**Conclusion:** no remaining direct blocker was found in the reviewed byte
retention scope after the corrections below. The normal path addresses the
previous HumanEval harvest gap: supported files from every `official_eval`
attempt, including failed attempts, and `.official-inspect-*` snapshots are
streamed into gzip without the old 2 MB text cap. The index records original
paths, raw byte counts/hashes, archive fingerprints and metadata bindings.

## Findings and verified corrections

- Self-certified index edits previously allowed files or metadata bindings to
  disappear while verification still succeeded. Verification now requires a
  caller-held index fingerprint, validates unique and constrained source/archive
  mappings, and rechecks the metrics-to-raw binding. The independent status or
  committed record carrying that fingerprint remains the trust anchor; this is
  not authentication against simultaneous replacement of all trusted records.
- Directory traversal errors and newly added attempts could be silently lost.
  Traversal errors are now retained; a final inventory comparison detects
  additions/removals, and every archived source file is rehashed to detect
  same-path changes after its copy. Observed changes produce partial retention.
- A failed exclusive-open could delete an existing archive. Publication now uses
  a private temporary file and atomic no-clobber link; cleanup removes only the
  temporary file. Repeated capture refuses to modify an existing evidence tree.
- Output parent symlinks could redirect writes. Directory-relative creation and
  opens now use `O_NOFOLLOW`, matching the source-side path protections.
- Retention I/O failures could prevent recording the scientific status. The
  harvest integration now records failed/partial retention and continues to
  retain the independent completion, judge and placement findings.
- Index/metrics verification previously hashed one open and parsed another.
  `_read_checked` now checks and parses the same byte buffer from one descriptor.
  Compressed archive verification also hashes, seeks and decompresses one open
  descriptor, checking its file state before and after.

The reviewed tests include large raw logs and failed attempts after removing
the source volume; missing/changed data and metadata; index mutation; directory
permission failures; additions and same-path changes during capture; output
symlinks; publication conflicts and repeated capture; index/inventory/gzip I/O
failures; single verified metadata reads; judge/placement separation; and the
legacy no-official-evidence path. Test execution and its final totals must be
reported separately by the main agent.

## Boundaries

`preserved` certifies this capture's supported byte inventory, not a valid
scientific result. It does not replace frozen-task PTB validation, model-byte
validation, judge review, receipt identity or placement checks. The archive is
not a standalone model bundle, and this review does not admit HumanEval to
compute-node execution. Unsupported files are explicit retention errors rather
than silently certified omissions.

Reviewed file SHA256 values:

| File | SHA256 |
| --- | --- |
| `awm/ptb_evidence_retention.py` | `441363be250e520e870ea47b8e6a44491fa0bec737323dfb159b63be734f75d3` |
| `awm/ptb_ops.py` | `f8b47689d6a43415e387886e41275f5e0c8ea3441c3eb55008c62189e5ee265c` |
| `tests/test_ptb_evidence_retention.py` | `55969077dbaabf48133fc5c372964f22a60d84190b73637303b874e0164e709f` |
