# Window04 prepared inputs — not an analysis decision

The current five NEW clean cells have machine-generated facts and timelines in
`prepared/`; [preparation.json](preparation.json) records tool/source identities,
successful exit codes and output hashes. The [accumulation roster](accumulation.json)
is still open at5/8, not a frozen analysis window. No reviewer or synthesis has
been dispatched and no protocol/queue decision is based on these summaries.

At dispatch, follow `skills/exp_protocol_meta/trace_review.md` and `metrics.md`:

- Treat these outputs as locators, not ground truth. For example, g01r04's
  cell-reader reports zero RL launches while the timeline emits a `first_rl`
  marker. Resolve the actual commands and timestamps rather than treating a
  first textual match as execution. Matched training counts are not a complete
  smoke/evaluation launch audit, and file-size `est_n` is not actual evaluated n.
- A waiting tool span is not automatically post-exit idle. Separate productive
  running time, composite commands, producer exit, later waiting and repair.
  Preserve raw card cost sums alongside an event-deduplicated interpretation.
- Permit each read-only reviewer to access its explicitly assigned bundles
  **and those cells' receipt-backed original `result_dir` paths in the roster**.
  Big Inspect JSONs can remain on the original data volume despite being
  excluded from the git bundle. Read actual samples/counts/config metadata when
  available; do not infer that evidence was lost merely from the bundle cap.
  This does not authorize scanning other queues or bypassing host trust checks.
- Identify protocol/control from the frozen manifest's paths/setup, not AWM
  bootstrap SHA or card count alone. Keep these old guard identities separate
  from the exact previously reviewed strict-guard cohort when tabulating.

Once at least eight NEW validator-clean cells accumulate, refresh/freeze the
roster to include all eligible unreviewed arrivals, generate the additional
inputs, and launch the prescribed local Opus5[1m] max reviewers plus synthesis.
Do not count old reports or these prepared files as completed deep review.
