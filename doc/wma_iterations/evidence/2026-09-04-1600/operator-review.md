# 16:00 UTC operator review

Ownership is OK on slurm2-a3nodesetondem-[2-3]: 16 allocated GPUs, 16 RUNNING,
49 safely routed PENDING. All 65 active jobs have null Slurm dependencies;
48 pending jobs report Priority and one Resources. Allocation is not measured
utilization. No replenishment or cancellation is needed. BFCL raw c53r02/03
(92182/92183) have started automatically as slots became free.

## Newly complete old control

c10r08 / 91432 passes the PTB validator and required automatic judges at
**73.0856709629%**. The original control and its extension now have eight
completed cells in total. Its agent clock is 08:42:29; Slurm allocation is
09:07:01 (9.1169 GPU-h); scientist reported cost is $33.74321075, excluding
judges. FAILED/2:0 remains the scheduler status despite valid final evidence,
as for previous old-cohort completions; it is not a scientific retry reason.

The frozen public AWM configuration is identical for the eight control and
eight WMA v0.2 cells (`ae46724`). The four manifests' contract objects are
identical after removing run_index. Both WMA manifests also pin private
`ae46724`; all completed attempts use PTB `62203e4`. The descriptive summary
is therefore confined to these matched Round 02 settings:

| Arm | n | Mean accuracy | Sample SD (pp) |
|---|---:|---:|---:|
| Protocol control, c10r01–08 | 8 | 74.9716% | 4.3408 |
| WMA v0.2, w10r01–08 | 8 | 72.3180% | 2.7382 |

WMA minus control is -2.6535 pp. This small unpaired sample does not establish
a causal harm or a promoted method. Original access flags remain visible;
these descriptive PTB scores are not a zero-leak promotion readout. No R1,
other runtime, candidate, or Opus4.8 outcomes are pooled. Full scores and
receipt/cell/manifest/spec/result paths are in `matched-r02-summary.json` and
`provenance.json`.

The unchanged terminal-verdict ledger for the eight WMA cells has **58**
verdicts, 36 scored, 22 suspected leaks; L0 hit .972, L1 hit 1.0, L2 coverage
.9 over 30 scorable cards, mean width .1185, width/noise 4.4546, L3 hit .704,
saved/wrongly-killed GPU-h both zero. It records $114.6868 for retained final
reviews, mean $1.9774 and 6.2591 wall minutes. This is not the complete request
history or total spend: previously reconstructed requests and overwritten
reviews use different denominators. No scanner flag or scorer changed.

## Newly failed BFCL raw baseline

c53r01 / 92181 has no final model or metrics, four validator issues and
general_anomaly; there is no scientific accuracy. The delegated bounded review
in `bfcl-failure.md` was checked against the final reply/result, killed task
records, earlier recovered OOM and the outer solve diagnostic. It confirms a
second instance of the same visible lifecycle sequence as GSM8K c51r01:
waiting for a background notification, normal CLI end_turn/completed, then
stopped wait tasks and no deliverable. The outer solve exits zero. The later
validator/evaluator failure explains batch FAILED/1:0, not the initial early
session return.

The general judge's claim of no CUDA/OOM event is contradicted by an earlier
recovered training OOM. Its stronger sole-harness/blameless-scientist attribution
is also unproven. The roughly 30-minute durations in both cases are a clue,
not evidence of an identified hard cutoff. The exact division among CLI
semantics, waiting behavior and wrapper lifetime remains open.

Scientist cost is $4.987571, recorded judges $3.83153925, combined $8.81911025
excluding GPU billing/unrecorded usage. Agent time is 00:31:15; allocated time
00:44:15 (0.7375 GPU-h). The original four BFCL raw replicates now comprise one
incomplete, two running and one pending. The failed GSM8K raw cell also remains
in its original denominator. Neither is replaced, assigned a fake zero score,
nor silently dropped from completion/yield reporting.

## Decision and next handoff

The highest-confidence new failure mechanism concerns scientist/CLI lifecycle,
not WMA policy: both affected cells are raw. A separate no-benchmark CPU marker
reproduction, with fixed cost/wall caps and background versus blocking cases,
is specified in the delegated note as a possible diagnostic. It has not been
run, and no continuation/prompt/runtime patch has been accepted or installed.
Any resulting common-runtime candidate requires its own preregistration and
actual-runtime acceptance; it cannot be silently applied to selected frozen
arms. No new skill candidate or promotion is justified by this check.

The inspected reconcile preview had exactly two harvests and 16 peeks, no
submit/cancel actions. Applying it archived both terminal attempts. Total PTB
validator/automatic-judge-clean completions are now 82; the Opus4.8 study has
zero complete and two incomplete terminal attempts. Existing manual semantic
flags remain separate from automatic judge-clean status. GPQA access remains
blocked by the dataset's permission requirement.

The existing hourly singleton is alive (PID 3591763). Four new clean cells are
unanalysed by the hook; the oldest was first seen at 10:39:06, so six-hour tail
eligibility begins at **16:39:06 UTC**, with the next hourly pass expected near
16:43 unless the eight-cell threshold is reached sooner. Current old ready
events already have completed operator handoffs. No duplicate analysis window
or timer was created.
