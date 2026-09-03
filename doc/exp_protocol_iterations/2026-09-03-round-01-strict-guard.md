# Round 01 strict guard — 2026-09-03

**Decision: the exact cohort passes its predeclared observed-no-harm gate; no promotion, no release.** Overall launch-order compliance remains unresolved. Detailed acceptance/rejection reasons are in the [planner decision](trace-reviews/round01-strict-guard-addendum/planner-decision.md).

## Variants

| label | frozen commit / protocol tree | difference |
|---|---|---|
| historical v3 | `eaf50919` / `08674f2c` | no session guard |
| strict guard | `4ae3d87c` / `189319d6` | session-end guidance, lifecycle pitfall and Stop hook |

This addendum is not a concurrent randomized arm comparison. Only g01s01–08 enter the strict cohort; old g01r01/02 and controls do not. g01s04 was already in Window03; these eight reports contain **seven new clean cells outside that window**, not eight more new observations.

## Cells

GSM8K · `google/gemma-3-4b-pt` · scientist `claude-opus-5[1m]`, high,1M ·10h/cell ·8 replicates · PTB `dcf5da0` · run_index2. Held-out task this round: none.

Manifest: `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml`. Immutable receipt: `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/formal-2026-09-02T204221.237369+0000.json`; jobs90791–90798. Bundle/status/provenance and raw-result paths are joined by cell in the [launch record](trace-reviews/round01-strict-guard-addendum/launch.json).

## Results

Full per-cell collect output: [guard-collect.csv](trace-reviews/round01-strict-guard-addendum/guard-collect.csv).

| variant | accuracy mean (min–max) | pitfalls_cost_h sum | n_locked_open sum | fields_filled mean |
|---|---|---|---|---|
| strict guard,8/8 clean | **0.720148 (0.695982–0.735406)** | **16.40h raw card sum**;14.30h after two retrospective duplicates, not exact trace cost | **0** | **1.0** |
| historical v3,14 clean/16 | 0.688563 (0.579985–0.776346) | see prior Round00 record; not recomputed here | prior window | prior window |

Guard sample sd0.013988, SEM0.004946;57 cards written/locked/closed. Historical v3 discovery was re-run with receipt-aware `awm ptb results`:14 eligible clean, no placement quarantine. Directly filtering old status files had wrongly yielded13 because legacy p00r05 lacks newer eligibility keys; absent fields must be resolved, not silently treated as false.

Three guard cards read by the planner (paths under the strict bundle):

- `g01s03/task/memory/cards/exp-05.yaml`: invalid-save attempt, repaired RFT and explicitly inconclusive small-n outcome kept distinct.
- `g01s08/task/memory/cards/exp-02.yaml`: OOM at step100, no checkpoint/measurement; hardware execution failure is not hypothesis falsification.
- `g01s04/task/memory/cards/exp-03.yaml`: eval-only decode intervention and a real diagnostic-data reference; do not label every optional reference fabricated.

## Trace review

All eight reports and the [unaltered synthesis](trace-reviews/round01-strict-guard-addendum/synthesis.raw.md) are retained. Local read-only Opus max reviewers: `73c154b4-944d-4e7d-bc60-5c192ed41207` (four), `35a6a6d1-b646-479b-ac76-8d26afcb5d0c` (three), prior Window03 `9081a22d-3a4d-455e-89ab-d3164d577a76` (g01s04). Synthesis: `79d04a31-ab3f-4fd5-9a20-4b3500f0ab52`. Extra Codex review covers ordering/duplicate costs and P1 observability; the planner also audited smoke scope.

Scores are set mainly by initial SFT/decode/recipe choices; observational differences do not prove protocol causality. D/B failures, E post-exit waits and H workarounds recur. The synthesis's all-producer11.17h total is provisional, not accepted as demonstrated recoverable savings. P1's stronger claims were narrowed by the additional audit.

## Directions

Retain E v2. Withdraw C v2's unstarted whole block because its primary threshold is already met by the baseline; keep evaluation-quality redesign pending. D/B/H become the first scientific wave, A/E v2 later with drift pairs. Do not register P1 v1 as written. Prioritize separate designs for full training/evaluation lock scope (#26) and prospective comparator workflow (#27). Prompt-distribution evidence remains investigation, not a universal recipe instruction. Ledger and planner decision preserve alternatives.

## Decision

The narrow predeclared gate passes:8/8 eligible clean results; no observed scientific work lost at session end;0 hook blocks,0 false blocks,0 locked-open cards; mean0.720148 exceeds the historical v3-minus0.03 floor0.658563. The planner independently checked all statuses, installed Stop-hook settings, absent block counters and57 concluded cards, and read the full terminal-evidence matrix.

This is **observed no harm**, not demonstrated protection: the hook never fired. Historical comparators do not prove score improvement; held-out confirmation is absent. Pre-lock evaluations and smoke runs are separate compliance defects requiring resolution, not a reason to erase PTB scores or silently redefine this gate. Ownership/native-isolation failures continue to forbid new submissions/releases.

## Change

No frozen scientist tree is changed by this decision. Reusable accounting and observation-time lessons go to meta metrics; C withdrawal changes planner queue intent only. Candidate E v2 remains separately frozen at `c6f11d8` / `ceb68549` with34 CPU tests and both manifest-check modes passing; it is not yet registered.

## Evidence

- [Ordering and duplicate-cost audit](trace-reviews/round01-strict-guard-addendum/g01s07-ordering-cost-audit.md).
- [Launch-scope audit](trace-reviews/round01-strict-guard-addendum/launch-scope-audit.md).
- [P1 predicate/observability audit](trace-reviews/round01-strict-guard-addendum/p1-observability-audit.md).
- [E retention/spec](../spec/2026-09-03-exp-protocol-round02-e2-process-wait.md).

## Next round and operational dependencies

At20:55 UTC, current checker: physical16/16, **registered17/16**, job90820 on a forbidden node, native reservation11 nodes. C jobs91054–91057 were cancelled from PENDING at21:10:24; `sacct` records zero elapsed and no start. At21:15:27 the operator verified **29 live JobHeldUser jobs**, all four cancellation receipts/terminal bundles and no remaining actions for the C-only queue. See [withdrawal audit](../../results/ptb/c-screen-withdrawal-20260903T211024Z.json). No running work was touched;90820 remains RUNNING and requires a separate user decision. Monitor PID2086813 remains live at3600s cadence; its21:00:54 sample has0/17 terminal jobs.

The first C reconcile missed terminal harvest because `_job_state` returned `CANCELLED by 0` verbatim. Fix `68253ee` normalizes Slurm decorations; the regression first reproduced the miss, then all111 PTB experiment/operations/results/gate/runtime tests passed. The C-only reconcile then harvested all four administrative withdrawals with null accuracy and `complete:false`; they add **zero** clean cells. Operator-only code changed, not the six shipped scientist paths.

Next: finish independent #26/#27 specifications, restore ownership/native isolation before any new receipt or release, and replace old E only after its new held receipt exists. Harvest every future terminal attempt, including90820 as quarantined evidence, then validate before counting clean cells. No held-out experiment, score promotion or queue release is implied by this record.
