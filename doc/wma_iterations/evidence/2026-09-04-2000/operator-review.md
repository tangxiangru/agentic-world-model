# 20:00 UTC operator review — second BFCL result and lifecycle recurrence

Ownership is OK. At20:01 UTC the subqueue has16/16 GPUs allocated,16RUNNING
and38 safely routed PENDING jobs on nodes2–3, with no bad routes or scheduler
dependencies. Thirty-seven pending jobs report Priority, one Resources. The
pending set contains37 original scientific jobs and validation-only92312; even
excluding validation, scientific reserve remains above32 and>8. No submission,
cancellation or replenishment is needed. Allocation is not utilization; direct
utilization remains unavailable under the recorded access limits.

The inspected reconcile preview contained three harvests and16 peeks, no
submit/cancel. Application archived the terminal cells and refreshed active
snapshots. Total scientific PTB-complete attempts are93; automatic-judge-clean
is91. The Opus4.8 study has4complete: BFCL P has2clean, BFCL R has1flagged,
GSM8K P has1flagged; two raw failures remain incomplete and a third HumanEval
raw failure is newly incomplete. Distinct validity strata stay separate.

## BFCL protocol-only P, now n=2 clean

c54r03 /92187 is clean at **94/100**, scientist cost$11.91053525, agent time
02:00:28, allocation02:11:05 (2.1847GPU-h); judge cost is unavailable. Its
required judges find a coherent delivery, exact Opus4.8/high/200k route, pinned
Gemma base, decontaminated public function-call data, no external generation API
or PTB lookup, and a verified final model. Three broad public-data recipes were
compared on the supplied 100-item ruler; the selected final checkpoint reruns at
.93 internally and scores .94 officially. Intermediate selection reads are not
independent final samples.

Together with c54r01=.91, P mean is **92.5%**, sample SD **2.1213pp**, n=2.
The two scientist costs sum$29.3500215; allocations sum4.7064GPU-h, excluding
judges. The remaining c54r02/r04 are running. Raw has no clean primary result,
so no P–R effect or protocol promotion can be computed. P's high score is
practical evidence, not a WMA result: this arm explicitly has no WMA.

## GSM8K P c52r02 — valid model, flagged sensitivity

c52r02 /92168 has a complete final model and official accuracy
**49.5830%**, but `general_anomaly` is confirmed. Its selected exp03 checkpoint
was valid; a later exp04 training run was still near step977/3456 when the
scientist said it would report after completion. A text-only end_turn followed,
the background task was killed, and exp04 never completed. The prior final model
and metrics remain valid, so this is a flagged complete sensitivity result,
not an incomplete zero. Specialized contamination/model/API/lookup judges are
clean. Scientist cost$35.4608325, agent time05:30:03, allocation05:49:58
(5.8328GPU-h); judges unavailable. Other GSM8K P cells remain running, and raw
has no clean comparator. No effect conclusion follows.

## HumanEval R c55r04 — incomplete

c55r04 /92192 ends after00:41:17 agent time and00:53:59 allocation (0.8997
GPU-h), scientist cost$6.68404925. The stable LoRA run was only around
221/2706 steps when ScheduleWakeup reported the dynamic loop unavailable; the
scientist then emitted a text-only waiting response. CLI returned
end_turn/completed with no API error, background tasks were killed, and no
final_model or metrics exist. It is incomplete/general_anomaly with no score.
Its data/model/API/lookup checks find no separate disallowed behavior; that does
not make a missing deliverable complete.

The operator independently checked both new lifecycle traces: end_turn,
terminal completed, null API error, killed background tasks, canonical model/
metrics presence for c52r02 and absence for c55r04, and unfinished training-log
markers. Hashes and booleans are in `lifecycle-crosscheck.json`; no question or
answer text is copied.

## Ranked decision

1. High confidence: five terminal attempts now share the observable one-shot
   lifecycle failure across GSM8K/BFCL/HumanEval and raw/protocol treatments:
   c51r01, c53r01, c53r04, c52r02 and c55r04. A text-only promise to wait ends
   the CLI session; background work is then killed. Some cells retain an earlier
   valid final model, others have no deliverable.
2. Attribution remains split between scientist waiting strategy and one-shot
   CLI/process semantics. This is not a hidden fixed30-minute cutoff: c52r02
   returned after5.5h, while other instances returned near30–160minutes.
3. The issue is common scientist/protocol runtime, not a WMA-skill cause. It can
   bias completion yield and leave unequal compute exposure. The raw baseline's
   failures are also practical baseline outcomes; changing only failed cells or
   assigning synthetic scores would break the frozen comparison.

No selective retry, scorer/guard change or in-place runtime edit is made. A
separate no-benchmark lifecycle reproduction and any symmetric runtime candidate
remain the correct next protocol/harness work; it must be preregistered and
accepted independently before future cohorts. Existing frozen cells continue.
The pending S0 smoke92312 checks model/broker/isolation compatibility, not
long-training lifecycle, and remains a real dependency for its four staged cells.

The hourly hook is alive. Its last pass counted six new clean cells after the
previous analyzed event, below the eight-cell trigger; the six-hour tail has not
matured. The prior Claude event has a completed corrected handoff. No duplicate
analysis is launched here. No WMA skill candidate or promotion is justified.

`new-results.json` preserves exact receipt→cell→manifest→spec→result paths,
source pins, judges and costs. `lifecycle-crosscheck.json` preserves direct
terminal evidence. Existing flags, failures and exclusions remain unchanged.
