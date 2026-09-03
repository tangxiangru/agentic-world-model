# WMA online Round 04 — scoped probes and candidate selection

Status: design review in progress; no Round 04 candidate launched.

The frozen design and gates are in
`doc/spec/2026-09-03-wma-round04-probe-selection.md`. Completed evidence and
independent trace audits are in `evidence/2026-09-03-w10/`.

At the 18:52 UTC scheduled check, ownership was OK and the WMA nodes had
16/16 GPUs allocated, with 17 safely routed pending cells. There were 48 clean
completed cells across distinct historical cohorts, unchanged from the prior
check. A–F had no completed result. The reconcile preview required only running
cell peeks and no new harvest or submission.

The previous event analysis was incomplete. Three bounded parallel specialist
reads completed access, uptake/levers and decision/harm diagnosis. Their
corrected findings were recorded in commit `9fccdde`. A separate bounded local
Claude Code `claude-opus-5` / max / ultracode follow-up was launched for the new
G/H/I/J design question. Its artifacts live in the original shared event's
`followup-design/` directory; this is a new preregistration question, not another
blind replay of the same broad event analysis. It is read-only and does not own
any launch, edit, commit or promotion.

Baseline remains v0.2 (`176f0a464986`), public/private runtime reference
`ae46724`. Independently justified candidate manifests will use four repeats
and frozen matching runtime bytes. The c10 tail and baseline/control extensions
are not needed to identify these prospective mechanisms; they remain needed
for their original formal comparison. No candidate is promoted by this record.

## Candidate archive preparation

The four reversible candidate texts are being frozen while the independent
design review runs. Freezing a candidate is not approval to launch it. Each is
one skill intervention against v0.2, and all non-skill private shipped bytes
are checked against `ae46724`. A preparation commit restores the old private
`lock.py` for these archives; after freezing, the branch restores the current
operator `lock.py` and the baseline skill. No existing manifest is changed.
This keeps the new archive's runtime identical to the comparison baseline
without discarding the operator's newer relock-history fix.
