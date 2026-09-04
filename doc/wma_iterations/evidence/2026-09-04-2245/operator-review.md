# Operator review — GSM8K first comparator and Claude handoff

This review closes hook event `20260904T214603Z-dc9ade9335` with corrections.
The event froze nine clean cells: BFCL protocol-only c54r01–04, GSM8K
protocol-only c52r03, and old H w14r01–04. The report completed successfully
with Claude Opus5/max/ultracode for $24.78373725. Its plan is advisory; it made
no repository, queue, or experiment change. The pre-review and both the raw
Claude response and plan are archived beside this file.

## Reproduced claims

The authoritative H ledger reproduces 25 retained verdicts, 14 with scoreable
truth, 11 original access flags, retained verdict cost $49.117, L2
width/noise 4.9453, L3 hit 0.600, and structural zeros for saved and wrongly
killed GPU hours. Direct reads reproduce that all 25 retained L3 answers are
`yes`. Of the 14 truth-bearing rows, four true rejects are all misses, six are
hits, and four are unscorable. The four misses are w14r01/exp-07,
w14r02/exp-06, and w14r04/exp-04 and exp-06.

The sidecar archives contain 38 response and processed records for 26 cards,
while only 25 final verdict files survive. The rejected
w14r02/exp-01 verdict is present and records $1.733 paid cost; current
`awm wma ledger` already reports it separately from the retained $49.117.
This confirms the accounting observation but does not authorize a post-hoc
scorer or ledger change.

The deployed cross-benchmark policy hash remains `17be8a23046a`. It contains
neither rejected H's C6 wording nor G's probe block, as intended: the new study
tests the separately frozen redesign. No H/G text is restored and no WMA skill
is edited from this report.

## New GSM8K evidence after the frozen event

The direct PTB refresh adds clean raw c51r02 and clean protocol c52r01/c52r04.
Every item has empty PTB issues and automatic judge flags, and the exact
receipt, cell, manifest, spec, source SHA and canonical result path are in
`new-results.json`.

Protocol-only primary clean scores are 0.6262319939, 0.5830174375 and
0.4283548143: mean **0.5458680819**, sample SD **0.1040379764**, n=3.
The lifecycle-flagged but complete c52r02=0.4958301744 remains sensitivity
only; all four give mean 0.5333586050 and sample SD 0.0885543998. Known
scientist spend is $118.783249 and one-H100 allocation is 30:32:49. Judge
costs are unavailable.

Raw c51r02 is the first clean raw comparator at **0.5481425322**, n=1,
scientist cost $11.883615 and one-H100 allocation 08:41:25. The descriptive
clean P−R difference is **−0.227445 percentage points**. Raw n=1 and the wide,
adaptively produced protocol spread do not identify a protocol effect. The two
bounded trace reports preserve the distinct recipes and do not read in-flight
trajectories.

Across the registered WMA scientific manifests the current readout is 100 PTB
complete and 97 automatically clean; the new Opus4.8 study contributes 11
complete and 8 clean. Automatic cleanliness does not erase the separate WMA
scope or HumanEval semantic exclusions.

## Decision and queue

Promotion remains `null`. H already fails 11 original scope flags; BFCL and
GSM8K P have no WMA; BFCL raw still has no clean comparator; the current GSM8K
raw comparator has only one clean result. Claude's N-A/N-B ideas remain
instrument proposals. N-B is partly stale because rejected paid verdicts are
already reported by the ledger. No new skill wording or runtime candidate is
accepted in this check.

At 23:05:26 UTC ownership is clean, all 16 WMA-subqueue H100s are allocated,
16 jobs are RUNNING and 30 are safely PENDING: 29 frozen scientific cells plus
validation-only job 92312. Twenty-nine pending jobs wait on priority and one on
resources; none has a scheduler dependency, bad route, unknown ownership, or
name mismatch. Allocation is confirmed; direct device utilization remains
unavailable from this operator environment. The reserve exceeds eight and the
24-job replenishment trigger has not fired, so adding redundant work would be
scientifically unjustified.

S0 w66r01–04 remain staged. Job 92312 must first validate the exact public and
private archives, model, context and isolation contract. No S0 science starts
before that verdict. The first wave remains frozen, and no running job is
cancelled or altered.
