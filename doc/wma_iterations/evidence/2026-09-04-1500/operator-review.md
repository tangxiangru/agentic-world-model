# 15:00 UTC operator review

The live ownership audit passes on the WMA subqueue: 16 allocated GPUs,
16 RUNNING, 51 safely routed PENDING, no scheduler dependencies. Fifty pending
jobs report Priority and one Resources. Allocation is not utilization; shared
monitor snapshots are incomplete and do not establish continuous full load.
No replenishment is required. The new GSM8K protocol c52r04 (92170) and BFCL raw
c53r01 (92181) have started automatically as capacity became available.

## G / w13r04 / job 91444 — complete, older Opus5 cohort

The result passes the PTB validator and all required automatic judges:
accuracy **0.7505686125852918**, standard error **0.011918265218445566**.
G coverage is 1/4; H is 0/4. A sample SD across G replicates cannot yet be
estimated. This is not a new score record, matched effect, or promotion.
The original v0.2 controls and runtime cohort remain the comparison; the
Opus4.8 study is separate. The original exclusions and manual audit flags on
other cells remain unchanged. This bounded check does not constitute a full
semantic access audit of the new G trace.

The unmodified authoritative ledger reproduces five scored cards for skill
`e4402ffa6bca`: L0/L1 hit 1.0, L2 coverage 0.5 over four scorable cards, mean
width 0.1276, width/noise 4.2534, L3 hit 0.5, and zero recorded saved/wrongly
killed GPU-hours. The scanner reports zero suspected leaks. All five retained
locks say delivered; their recorded waits total 1796.9 seconds (0.4991 GPU-h).
Delivery alone does not prove every executed version matched its lock.
By-type rows overlap and must not be summed as independent verdicts.

Ledger WMA cost is $9.9815; mean review wall time is 6.2617 minutes. The
scientist CLI reports $42.52521025, excluding judges. Agent time is 06:22:09;
Slurm allocation is 06:46:37 (6.7769 GPU-h). Slurm nevertheless records
FAILED/2:0; the output ends with PTB COMPLETE FLOW PASSED. Preserve both facts;
the exact batch-script exit-2 cause is not established by the bounded tail.

## R / c51r01 / job 92163 — incomplete, new Opus4.8 cohort

The validator rejects this attempt: final_model/config.json, weights and
metrics are missing. There is no scientific accuracy. The general judge flags
anomaly; the other required judges do not flag contamination/model/API/lookup.
Keep this attempt and its cost in the original four-replicate denominator.
The other three raw GSM8K replicates remain running.

Direct evidence confirms the following sequence, independently of the judge's
interpretation (see the numbered `raw-termination-excerpt.txt`):

- At parsed trace lines 2735–2736, the scientist says its background monitor is
  running and it will wait for a completion notification.
- Lines 2738–2789 contain a successful CLI result with stop_reason=end_turn,
  terminal_reason=completed, api_error_status=null and that same final reply.
- Lines 2830–2912 record background tasks killed/stopped at 14:18:37 UTC.
- The final evaluation fails on the absent model; these downstream vLLM errors
  are not evidence that an evaluator outage caused the missing deliverable.

The frozen raw launcher at PTB `0bb448c`,
`agents/claude_vertex_high_200k/solve.sh:16`, runs a single `claude --print`
session; it has no automatic continuation loop. The confirmed failure is a
session-lifecycle/background-wait interaction while work was unfinished. The
judge's stronger claim that the harness alone is responsible and the scientist
is blameless is **not established**. No API error or Slurm timeout is reported
for this exit. The precise CLI-internal cause remains unconfirmed.

The scientist reports $3.50933425, 67 turns, duration_ms=1804613. The wrapper
records 00:31:17 agent time; allocation is 00:43:28 (0.7244 GPU-h), FAILED/1:0.
These are distinct clocks and exclude subsequent judge cost from the CLI cost.

## Ranked causes and decision

1. High-confidence operational failure: end_turn occurred before a deliverable
   while background work remained. This concerns common scientist/CLI lifecycle,
   not the WMA skill (the failed arm has no WMA).
2. The exact division between scientist waiting behavior and CLI completion
   semantics requires a bounded reproduction; it is not yet a proven external
   infrastructure fault eligible for selective replacement.
3. G's isolated score and broad L2 widths do not support a new skill edit or a
   claim of improved action selection. Retain the preregistered wave.

No automatic retry, skill edit, runtime patch, scorer change or queue mutation
is made. Do not retrofit continuation into the frozen remaining arms or replace
only failed raw outcomes. If a lifecycle fix is proposed, it needs a separately
preregistered common-runtime comparison and real acceptance; original failure
and cost accounting must remain visible. Track this failure category across
all five arms without reading in-flight scores as a comparison result.

The reconcile preview was inspected: exactly two harvests plus 16 peeks and no
submit/cancel actions. Application completed both bundles. Full provenance is
in `provenance.json`, validation output in `results.json`, and unchanged ledgers
in `g-ledger*.md`. A requested bounded delegate was unavailable because the
collaboration tool returned “agent thread limit reached”; the operator performed
the bounded checks directly and makes no claim of independent peer review.

The hourly singleton is alive (PID 3591763); its last pass was 14:42 UTC. There
are now three clean cells beyond the last analysis watermark (the two earlier
control tails plus G/w13r04), below the eight-cell trigger and four-cell tail
minimum. No duplicate Claude analysis or old-event handoff was created.
