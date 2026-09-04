# Operator review — GSM8K raw n3 and failed WMA delivery

This check inspected the reconcile preview before harvesting two terminal
cells, then repeated the preview and harvested a third: GSM8K raw c51r04 and
c51r03 are complete and clean; GSM8K single-WMA w57r01 is incomplete and
flagged. The exact result queries, receipt paths, bounded trace reports and
queue snapshot are preserved beside this file.

## Raw GSM8K result

c51r04/job92166 is validator- and judge-clean at 878/1319 = **66.5656%**.
It used scientist Opus4.8/high/200k, cost $26.44032375, agent time09:19:39 and
one-H100 allocation09:38:18. Its 558 monitor samples average70.88% GPU
utilization, with80.82% nonzero; this historical per-cell utilization is kept
separate from allocated GPU-hours and does not establish current live-device
utilization.

c51r03/job92165 is validator- and judge-clean at **42.4564%**, scientist cost
$35.21025, agent time09:33:44 and one-H100 allocation09:53:11. Its 572 monitor
samples average83.84% utilization, with91.61% nonzero. Judge costs for the raw
cells are unavailable.

The clean raw arm is now 54.8143/42.4564/66.5656 = **54.6121% ±12.0559pp
sample SD (n3)**. The protocol-only primary is 62.6232/58.3017/42.8355 =
**54.5868% ±10.4038pp (n3)**. Descriptive P−R is **−0.0253pp**, with a Welch
standard error of9.1939pp; this is effectively an unresolved tie, not evidence
of benefit or harm. c51r01 remains incomplete, not zero. The single 66.57%
maximum is not a treatment effect.

Known clean-raw scientist spend is $73.53418875, allocation28.2150GPU-h and
agent time27:13:55. Including incomplete c51r01 brings known raw scientist
spend to $77.043523 and allocation to28.9394GPU-h. No selective retry is made.

## w57r01 failure and WMA compliance

w57r01/job92198 is Slurm FAILED with exit1:0 and PTB incomplete. It has no
`final_model`, weights, metrics or official accuracy. A previously measured
exp-01 dev score0.50 was never packaged and is not a result. The main session
ended at exp-03 step approximately25/1218 while explicitly waiting for
background training; the background process was killed five seconds later.
This is the eighth recorded background-wait/end-turn lifecycle instance.

All three WMA reviews failed with the same Opus4.8 input-token quota 429 and
produced no verdict. Lock waits were190.2/195.2/190.2 seconds; measured call
wall totals9.411 minutes and reported API cost is zero. The scientist recorded
`proceed` after each failure, but a fallback action is not a WMA verdict. Under
the standing online WMA gate this cell has **0/3 delivered verdicts** and cannot
represent the current WMA skill even if it had produced a model.

The cell used3.0078 allocated H100-hours and $16.58684725 scientist cost; four
judges cost$3.42892375, for $20.015771 known API spend. General anomaly is true;
contamination, model, API and PTB-lookup judges are otherwise clean. It stays a
distinct failed/incomplete attempt and is not assigned a synthetic score.

## Ranked causes and next action

1. Provider quota / enforcement: terminal evidence proves three Opus4.8 WMA
   429 failures. Ten measurement snapshots in six running cells have the same
   one-turn, zero-dollar, approximately three-minute fingerprint; because they
   are in-flight, they are strong health signals and not terminal conclusions.
2. Harness lifecycle: text-only end-turn again killed intended background work
   and left no deliverable.
3. Score variance: raw and protocol clean means differ by0.025pp while both
   arms have10–12pp sample SD, so the current control comparison cannot resolve
   a small effect.

No WMA skill edit follows from an API quota or enforcement failure. Candidate
V1 is preregistered separately as a protocol/harness-only delivery gate: no
matching delivered verdict means no launch. It names its compliance metric,
falsification and leak/cost/PTB guards; no implementation or job is started.
The old validation92312 cannot validate V1 because it freezes the pre-V1
runtime. Formal S0 must not launch through the observed failed-review fallback.

## Queue and decision

At23:44 UTC ownership and routes are clean,16/16 H100s are allocated,16 jobs
are RUNNING and26 are safely PENDING (25 priority, one resource). Normal
backfill has started GSM8K joint w59r01–03. Pending remains above the24-job
replenishment trigger and well above the hard floor; no redundant experiment is
submitted. All observed active receipts have null scheduler dependencies.

Live GPU allocation is confirmed; direct live utilization remains unavailable
under the recorded access limitation. The hourly hook is alive. Six clean cells
have accumulated after the last frozen event, below the eight-cell trigger and
well before the six-hour tail rule. No duplicate Claude analysis starts.

Totals are now103 PTB-complete/100 automatically clean, with14/11 in the new
Opus4.8 study. Promotion remains None. No running or pending job is cancelled
or altered, and no scorer, judge, guard or frozen treatment is changed.
