# 19:00 UTC operator review — first BFCL results

Ownership is OK on slurm2-a3nodesetondem-[2-3]. At 19:00 UTC there are
16 allocated GPUs, 16 RUNNING and 41 safely routed PENDING jobs, with no bad
routes or scheduler dependencies. Forty pending jobs report Priority and one
Resources. The pending set includes validation-only S0 job92312; even excluding
it,40 scientific pending jobs remain above the32 target and>8 floor. No
replenishment or cancellation is needed. Allocation is not utilization; direct
utilization remains unavailable under the previously recorded access limits.

The inspected reconcile preview contained two harvests and16 peeks, with no
submit/cancel actions. Application archived both terminal BFCL attempts and
refreshed active snapshots. Total scientific PTB-complete attempts are91;
90 pass the validator and automatic judges with no flags. The new Opus4.8 study
has two complete attempts: one clean protocol-only result and one flagged raw
result. All incomplete/running cells retain their original denominator.

## P / c54r01 / job92185 — clean primary result

This is the first clean result in the new cross-benchmark study: **91/100 BFCL
exec_simple items**. The receipt freezes top225bd584, PTB e62036f, Opus4.8/high/
200k, Gemma3-4b-pt cc012e0a and the original containers. Runtime provenance
confirms the resolved scientist model/context and WMA disabled, as required for
the protocol-only arm. The one frozen experiment lock records `wma=not_attached`,
zero WMA wait, explicit single decision-mode fingerprint, and script/data hashes.

Observed recipe: a base comparator of .04, then one full SFT on 25,503 selected
xLAM examples built from 26,765 decontaminated rows. The trained checkpoint read
.92 on the 100-item ruler; the copied/verified final_model rerun read **.91**,
which is the official result. Required contamination/model/API/lookup/general
judges are clean. This supports successful delivery and a large within-cell
change from base; it does not estimate a protocol effect.

Scientist cost is$17.43948625, agent time02:19:31, allocation02:31:18
(2.5217GPU-h). Judge costs are unavailable in the harvested bundle, not zero.
The other three P replicates remain running, so mean/SD and formal P–R contrast
are not estimable.

## R / c53r04 / job92184 — complete but flagged sensitivity

The raw cell's final_model is valid and scores **92/100**, while the general
judge flags a premature end. The scientist had preserved its best measured ft_r1
checkpoint, but then launched an unfinished round3, confirmed GPU work in
progress and ended with a promise to wait for a background notification. The
one-shot CLI returned end_turn/completed; its background waiter/training was
then killed. No round3 checkpoint/evaluation exists. This repeats the lifecycle
pattern seen in raw GSM8K c51r01 and BFCL c53r01, although unlike those two this
cell had already materialized a valid best checkpoint.

The final ft_r1 training set passes the recorded decontamination check and the
specialized contamination/model/API/lookup judges do not flag. A discarded r2
synthetic dataset did echo viewed test styles/items, but r2 was not promoted to
final_model; no score-causality claim follows. General anomaly remains decisive
for the preregistered clean primary readout. Keep .92 only as a flagged
sensitivity result; do not silently compare it as a clean raw baseline or use
the one-point P/R difference as a treatment effect.

Scientist cost is$12.70029250, agent time02:37:35, allocation02:50:24
(2.84GPU-h). Judge costs are unavailable. Raw coverage is one flagged complete,
one prior incomplete failure and two running. Its primary clean mean/SD remain
undefined.

## Decision

The clean P result is promising practical evidence for Opus4.8 on BFCL, but n=1
and absent clean R comparator prevent a method conclusion. One BFCL item equals
one percentage point; selecting .92 over .91 would be maximization on two
different cells and validity strata. Wait for the three P and two running R
replicates, report failures and flagged cells, then compute the frozen P–R
contrast. No skill, protocol, scorer, guard, retry or additional scientific
submission is justified by this readout.

S0 production validation92312 is still safely PENDING; its four formal cells
remain staged. The existing hourly hook is alive. Its last pass records five
new clean cells beyond the previous event, below the eight-cell trigger and
far short of the six-hour tail wait. The latest seven-cell Claude report already
has a corrected operator handoff, so no analysis is duplicated here.

`bfcl-first-results.json` preserves receipt→cell→manifest→spec→result, frozen
configuration, scores, judges, locks and cost fields. The bounded raw anomaly
review is retained separately. Original evidence and flags are unchanged.
