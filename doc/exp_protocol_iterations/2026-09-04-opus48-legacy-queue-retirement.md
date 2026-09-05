# Retire superseded whole held blocks for the approved Opus4.8 study

Planner decision2026-09-04; execution still requires fresh exact-job checks.
The user authorized pruning unnecessary pending work and approved the40-cell
[cross-benchmark study](../spec/2026-09-04-exp-protocol-opus48-cross-benchmark.md).
The16 new GSM8K receipts already provide independently justified held work.
This decision retires old configurations, not hypotheses or already-observed
results; no new Opus4.8 session substitutes statistically for an old Opus5 repeat.

## Entire receipt membership selected

| Block and immutable receipt | Exact job IDs | Decision rationale |
|---|---|---|
| [A v2](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-a-decode-x4-v2/formal-2026-09-03T024605.449036+0000.json) |91046–91049|The approved package/model study does not fund the older Opus5 single-item decode-latency screen. E does not reproduce or prove A's original timed endpoint.|
| [B v2](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-b-vllm-sampling-x4-v2/formal-2026-09-03T024630.343660+0000.json) |91050–91053|Sampling/serving engineering enters E6; separate old single-item attribution is not a prerequisite for the new package contrast.|
| [drift A](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-guard-drift-a-x2-v2/formal-2026-09-03T024718.653023+0000.json) |91058–91059|Companion to the retired old-generation wave; no independent drift top-up is funded.|
| [H](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-h-eval-only-data-x4-v1/formal-2026-09-03T024833.008855+0000.json) |91068–91071|Honest non-training data applicability is in E2; retire the old standalone configuration.|
| [drift B](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r02-guard-drift-b-x2-v1/formal-2026-09-03T024856.866361+0000.json) |91072–91073|Companion to the retired old later wave, not required for Opus4.8 comparisons.|
| [strict-control repair](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1-v1/formal-2026-09-04T052527.042436+0000.json) |91965|Explicitly stop completing the old strict8 cohort; its repair spec permits withdrawal after a later scientific decision. Preserve strict7 plus placement sensitivity, not a fictional repaired8.|

Main and an informed reviewer independently checked metadata: six complete
receipts,17 unique jobs, no additional members, matching jobs/cells, old
Opus5/high/1M/GSM8K/10h and frozen ondem0–1. The later approved package study
supersedes Window04's earlier B/H priority and A hold; it does not retroactively
declare any screen effective, ineffective or saturated.

The mixed baseline receipt is excluded:90826–90830 remain untouched alongside
its3 completed attempts. Already completed cells, all raw evidence, frozen
manifests/methods and receipt memberships stay unchanged. No running, configuring
or foreign job may be cancelled. A future revived question needs a new immutable
manifest/receipt and its matching old-generation controls, not a new-model proxy.

## Execution contract

Change only these six queue entries to want:cancelled with the above rationale.
Before applying: fresh registry-aware OWNERSHIP OK; exactly these17 jobs still
PENDING/JobHeldUser, matching receipt names/nodes, with no recorded start/runtime;
confirm the16 new jobs92125–92140 remain useful held on their frozen nodes.
The normal operator's pending-only cancellation requests and controller
confirmation are mandatory. No hand-written scheduler submission or release.
Inspect the dry reconciliation and permit only this exact retirement and its
receipt-backed harvest; stop on any unexpected mutation.

If all gates hold:38 physical holds →21, comprising16 useful new cells plus5
untouched mixed-baseline holds. The useful floor remains16≥8. Cancellation
records append to existing receipts; collect every administrative terminal,
including any unexpected retained attempt. They add zero clean model results
unless actual independent validator/judge evidence says otherwise.

Keep the live hourly detector and cumulative watched IDs through its next event.
Administrative terminals may wake it but never trigger a clean-cell synthesis.
After its real threshold event and full harvest, rearm only remaining relevant
jobs, preserving old state and zero new Opus4.8 clean count. No native reservation
change, release exception, GPQA access or node-login authority is granted here.

Preflight15:12:35 UTC passed: OWNERSHIP OK, all17 exact retirement jobs and
all16 retained new jobs are JobHeldUser, zero runtime, StartTime Unknown,
zero restarts, exact receipt names and ReqNodeList on ondem0–1. GPUs allocated0.
[Raw controller/receipt evidence](analysis-2026-09-04-opus48-onboarding/legacy-retirement-preflight.json).
No mutation occurred in that check; operator apply must still recheck PENDING.

## Execution completed

Planner/source decisionff9143e was pushed before operator execution. The clean
shared operator clone was fast-forwarded; its dry plan contained exactly17
cancellations and nothing else. Immediate15:16:47 preflight reconfirmed all33
target/retained jobs and ownership. Normal apply confirmed17 pending-only
cancellations, then the next exact17-action plan harvested them all. All have
complete=false, eligible=false, accuracy=null, result directory not found.
Commit4cd1dbd records the six appended cancellation lists,17 status bundles and
ops log; pushed and fast-forwarded back to the planner. No existing result was
deleted. Old configurations remain reproducible through new future receipts.

[Postflight15:22:09](analysis-2026-09-04-opus48-onboarding/legacy-retirement-postflight.json)
confirms all17 CANCELLED, no allocated resources, zero runtime and accounting
Start=None/ElapsedRaw=0. All16 new jobs remain JobHeldUser with exact names/nodes.
Current inventory21 physical holds,16 useful; ownership OK,0/16 allocated.
Slurm controller StartTime becomes the cancellation timestamp (equal EndTime)
even for these never-started jobs; do not confuse that with actual allocation.
The initial postflight's Unknown-only assertion exposed this difference; the
corrected check retains both controller and independent accounting evidence.

The historical console suffix “clean” on these incomplete harvests meant only
an empty flags list, not judge-clean completion. Status JSON never marked them
complete. The logger is corrected prospectively to say judges-unverified for
incomplete/flagless results; existing logs remain unaltered.

Monitor3564003 wrote its15:16:40 tick just before cancellation, still0/38 seen
terminal. Keep it live and unchanged until its next16:16:40 event; the17 known
administrative terminals are already harvested but do not create clean evidence.
After the real detector event, archive its state and rearm the remaining21 jobs.
