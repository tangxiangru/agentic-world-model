# 21:45 UTC operator review — BFCL protocol arm complete

Ownership is OK. At21:45 UTC the subqueue has16/16 GPUs allocated,16RUNNING
and33 safely routed PENDING jobs on nodes2–3, with no bad routes or scheduler
dependencies. Thirty-two pending jobs report Priority and one Resources. The
pending count is32 scientific jobs plus validation-only92312, so scientific
reserve alone meets the planning target and remains above>8. No replenishment,
submission or cancellation is needed. Allocation is not utilization; direct
utilization remains unavailable under the recorded access constraints.

The inspected reconcile preview contained two harvests and18 peeks, no
submit/cancel. Application archived both clean terminal attempts. Scientific
PTB-complete/automatic-clean totals are97/94. The Opus4.8 study has8complete,
5clean. GSM8K single-WMA w57r01–04 now all RUNNING with source31b854bb, skill
17be8a23046a, Opus4.8/high and blocking single mode.

## BFCL protocol-only P complete

c54r04 /92188 is clean at **90/100**, scientist cost$20.02233275, agent time
03:00:37, allocation03:11:11 (3.1864GPU-h); judge cost unavailable. All required
validator/judge and frozen model/source checks pass. P's four clean scores are
**91/88/94/90**, mean **90.75%**, sample SD **2.50pp**, range88–94,n=4.
Known scientist spend is$70.021497 and allocation12.1992GPU-h, excluding judges.

BFCL raw remains zero clean: c53r01 incomplete, c53r02 running, and c53r03/r04
complete but `general_anomaly` flagged at91/92. The raw sensitivity mean91.5%
cannot be compared with clean P. P–R primary is undefined; BFCL P's stable high
absolute score is practical protocol-arm evidence, not a causal protocol gain or
WMA result. No AIME/held-out or promotion action follows.

## GSM8K protocol-only P first clean result

c52r03 /92169 is validator/automatic-judge-clean at **58.3017%**. Scientist
cost$25.333925, agent time07:26:18 and allocation07:45:48 (7.7633GPU-h);
judge cost unavailable. The arm has one clean result, one flagged complete
c52r02=49.5830%, and two running cells. Raw GSM8K still has zero clean results.
Do not combine validity strata, compute an arm SD from one clean cell, or compare
this score with old Opus5 cohorts whose scientist/context/runtime differ.

## Decision and analysis handoff

No skill, protocol implementation, scorer, guard, retry or additional science
is changed/submitted. Await the remaining frozen replicates and first WMA-arm
outcomes. S0 validation92312 remains safely PENDING; its four formal cells stay
staged behind exact acceptance. HumanEval semantic-guard implementation remains
separate from this readout and from WMA policy.

The hourly hook had seven new clean cells at20:45; these two clean completions
raise the unanalysed window above the eight-cell trigger. The existing singleton
will freeze and launch one Opus5/max/ultracode analysis on its next pass. Do not
manually duplicate that window. The previous report already has a corrected
operator handoff.

`new-results.json` preserves receipt→cell→manifest→spec→result, exact scores,
source pins, judges, costs and timing. Primary/sensitivity exclusions and all
previous failures remain unchanged.
