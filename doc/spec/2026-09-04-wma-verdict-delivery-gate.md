# WMA verdict delivery gate

Status: preregistered design only. No implementation, validation receipt, or
scientific job exists for this candidate.

## Problem and frozen evidence

The current WMA online contract requires a delivered verdict before a card may
launch. Terminal Opus4.8 single-WMA cell w57r01 violates that contract in a
directly observable way: exp-01, exp-02 and exp-03 each waited for the WMA,
received Vertex 429 `RESOURCE_EXHAUSTED`, wrote `lock.wma.state=failed` with no
verdict, and then recorded an explicit `proceed` action. All three experiments
started. The cell later ended incomplete for a separate background-lifecycle
failure and has no official score or model.

The terminal evidence is three failed reviews, zero delivered verdicts and
575.6 seconds of lock wait. A bounded health audit of the contemporaneous
in-flight mirrors found ten additional one-turn, zero-dollar, approximately
three-minute review measurements with the same 429-like fingerprint in six
running cells. Those ten are operational signals only, not terminal failures
or scientific results.

## Candidate V1 — fail closed without a delivered verdict

Make one protocol/harness behavior change: a WMA-bearing card is not runnable
unless its matching lock records `wma.state=delivered`, a non-null verdict path,
and the expected plan/lock/file/treatment fingerprint. `failed`, timeout,
missing, stale or mismatched review states cannot be converted to `proceed`.
This candidate changes no WMA skill, scorer, judge, result, or existing receipt.

The targeted mechanism is enforcement. It does not solve provider quota; it
prevents a quota failure from silently turning a WMA treatment into a no-WMA
run bearing the WMA arm label.

## Preregistered readout

Primary metric: delivered-verdict-before-launch compliance. The acceptance
target is 100% over a four-case validation matrix, with zero launch records for
every non-delivered case and exactly one launch for a matching delivered case.

The four negative cases are review failure/429, timeout, missing verdict, and
fingerprint mismatch. A positive live acceptance must additionally obtain a
real Opus4.8 verdict with the frozen model, high effort, 200k context and
isolation contract before any future GPU science is submitted.

Falsification: reject V1 if any negative case can create a launch/action marked
`proceed`, if a delivered case can launch under a mismatched fingerprint, or if
the positive live acceptance cannot produce a real verdict. A blocked launch
under continuing 429 is correct enforcement but does not establish production
readiness.

Guards:

- leak: zero outside the existing WMA isolation fence;
- cost: no GPU process starts before delivery, and per-card review wait may not
  exceed 22.5 minutes, 1.5× the existing 15-minute WMA wall budget;
- PTB: no historical status, score, judge flag or receipt is rewritten; current
  running/pending cells remain their frozen cohort;
- separation: implementation and validation are protocol/harness work only;
  `skills/wma/` remains byte-identical;
- promotion: V1 can only be accepted as an enforcement fix. It cannot promote
  a WMA skill or establish benchmark benefit.

## Dependencies and launch rule

The existing S0 validation job 92312 freezes the pre-V1 runtime. It remains
useful evidence about provider availability and the legacy private archive, but
it cannot validate this new enforcement behavior. No new WMA-bearing science,
including formal S0, may be submitted through the failed-review `proceed`
fallback. After an implementation exists, it requires its own validation and a
matched-comparison decision; changing only S0 would break the frozen S0 versus
current-policy comparison.
