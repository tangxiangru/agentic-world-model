# D v1 whole-block withdrawal — 2026-09-04

Status: **all4 unstarted jobs cancelled and harvested**. This is an independent known-defect disposition, not acceptance of the unfinished revised synthesis's D2 design.

## Decision and exact targets

Withdraw all four **unstarted** D v1 cells d02r01–04, jobs **91060,91061,91062,91063**, through their [immutable receipt](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-d-parent-config-x4-v1/formal-2026-09-03T024742.743201+0000.json) and existing queue entry. The [manifest](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-d-parent-config-x4.yaml), frozen8332917 candidate and original job membership remain unchanged. No running work is authorized for cancellation.

The [exact frozen-code audit](trace-reviews/window04-local/d-scope/report.md) and planner CPU replay confirm false blocking of pure evaluations and verified in-code save repairs. A4-cell v1 GPU block is not needed to rediscover this known scope defect, and its stock-only false-positive readout would miss the relevant class. Preserve the useful save-safety direction and true Trainer/merge-save failure evidence; a replacement needs its own design/tests/immutable manifest. The first synthesis's family-only D2 proposal has unresolved false-positive/guardrail contradictions and is **not** accepted by cancelling v1.

Alternatives considered: keeping v1 held indefinitely inflates the visible backlog without making it releasable; running it despite known scope defects consumes four10h budgets without answering the corrected question; silently rewriting its manifest would destroy provenance. Whole-block withdrawal preserves all history and is independent of the remaining synthesis choices. The existing “replacement receipt first” rule applies to stale E, not this D decision; E is not touched here.

## Pre-action evidence and buffer

At06:11:06 UTC, the current registry-aware view is OWNERSHIP OK,30 actual JobHeldUser,0 allocated GPUs, no unknown/name mismatches. All four D job names match the exact receipt and all four are PENDING(JobHeldUser). `sacct -nX` additionally reports elapsed0 and Start/End Unknown for every target. No cell is selected by outcome: none ran. Recheck before apply.

Removing D4 leaves26 actual held, including staleE4; **22 other independently specified held cells remain**, well above8. D was already excluded from the useful22 and the hourly detector, so withdrawal does not reduce that useful buffer or alter its cumulative22-ID watch set. There is no release/submission in this decision.

## Cancellation-path protection

Before issuing this withdrawal, the operator now includes `--ctld --state=PENDING` with the exact receipt job ID, in addition to its pre-read PENDING guard. Installed Slurm25.05.2 advertises both flags. [SchedMD's scancel contract](https://slurm.schedmd.com/scancel.html) documents controller-side filtered requests and the pending-only state restriction. No user/partition-wide cancellation is used.

An exit0 from a filtered request is not itself confirmation. The operator re-reads the canonical state and records success in `cancellations` only if it observes CANCELLED; RUNNING/PENDING/UNKNOWN/FAILED outcomes raise an unconfirmed result and leave the receipt unchanged. Later terminal reconciliation still harvests every outcome. Four CPU race/unconfirmed cases exercise that distinction, and both operator/launcher suites pass **100 tests**. These tests are not a real-job cancellation result or a native-isolation fix. No scientist runtime path or PTB pin is changed.

## Execution and harvest

At06:19:39 the final ownership/job-name/hold gate passed again; all4 had elapsed0 and Start Unknown. Normal `awm ptb reconcile --apply` confirmed all4 CANCELLED at06:19:41, with `state_before:PENDING`, `state_after:CANCELLED` and `pending_only:true` appended to the tracked receipt. `sacct` independently reports all4 `CANCELLED by0`,elapsed0,Start None. The original job membership/source fields remain unchanged. No running job, unrelated ID, model output or experiment result was removed.

The next dry reconciliation contained exactly4 harvests; apply created all4 administrative status records. Each is `slurm_state:CANCELLED`, `complete:false`, `eligible:false`, `accuracy:null`, with `result directory not found`. These are withdrawn unstarted plans, not failed scientist trajectories or new clean cells. At06:21:03 the live registry view is OWNERSHIP OK,26 actual held,0 running/allocated; excluding staleE4 leaves22 usable. The live hourly monitor's22 IDs are unchanged because D4 was already excluded.

One pre-action issue was corrected before any cancellation: the control-repair verification JSON had been placed at its batch root, where all JSONs are discovered as receipts. It was relocated to `audit/` and all references updated; dry reconciliation then returned exactly the intended4 cancellations. This preserves strict receipt parsing and all evidence. No malformed receipt was ignored to force the action through.
