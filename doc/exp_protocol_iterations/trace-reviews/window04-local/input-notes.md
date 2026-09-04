# Window04 frozen inputs — deep review in progress, not a decision

The window is now frozen at14 NEW clean cells plus one explicitly historical
baseline calibration. [roster.json](roster.json) is the current authority and
[inputs.json](inputs.json) records all prepared facts/timelines and hashes.
The earlier `accumulation.json` and `preparation.json` preserve the five-cell
checkpoint only; they are superseded, not the current dispatch gate. Reviewers
are being dispatched, but no synthesis or protocol/queue decision is complete.

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

The minimum-eight gate is satisfied by all14 NEW eligible arrivals at the
fixed04:00 trigger. The three then-running jobs are tracked separately; do not
silently add late arrivals or calibration to this frozen denominator. Complete
all assigned reports, then use an independent local Opus5[1m] max synthesis.
Do not count prepared files as completed deep review.
