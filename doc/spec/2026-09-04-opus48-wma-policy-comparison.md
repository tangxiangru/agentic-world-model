# Opus4.8 matched WMA policy comparison

Status: preregistered before any result from the new Opus4.8 study; preparing
an archived legacy policy and its required production acceptance. Formal cells
remain staged until that exact public/private/model acceptance passes.

## Question and independent basis

The user requested newer methods and a GSM8K comparison, alongside other
benchmarks. The first 60-cell study compares raw PTB, protocol, single-card,
multi-self and joint modes under the new policy/runtime. It estimates practical
method contrasts but does not isolate old versus new WMA policy. Add exactly
one four-cell legacy-policy setting on the same new runtime to answer that
question. This is a reference-policy comparison, not a new optimized candidate
or an attempt to rehabilitate old scope violations.

The operator accepts the useful contrast suggested by Claude's advisory N5,
while rejecting an artificial dependence on H or unrelated scientific tails.
It depends only on its own frozen source and technical acceptance. No current
Opus4.8 outcome was used to select this comparator; none is complete at design.

## Frozen treatments

- Existing S: `wma-crossbench-opus48-r05-gsm8k-single-x4`, w57r01–04; public
  and private `31b854bbc5e1f7f66685a8ec0d43845a6c2472f1`, skill `17be8a23046a`.
- Added S0: `wma-crossbench-opus48-r06-gsm8k-legacy-v02-x4`, w66r01–04.
  Public source stays `31b854bb`; private executable paths must be byte-identical
  to `31b854bb`, with only the three `skills/wma/` files restored byte-for-byte
  from `ae4672487cc16f1d75921dcaab85498d7adc938e`, skill `176f0a464986`.
  The archive commit and resulting hashes are written into the record/manifest
  before technical validation or scientific submission. Restore the current
  operator skill after archiving; never change the existing S receipts.
- Both use single-card decision mode, blocking WMA review, scientist and WMA
  `claude-opus-4-8` / high, 200000 scientist context, 10h agent budget, one H100,
  16 CPU, 128GiB memory, 400GiB scratch, identical curated train history and WMA
  budget `cpu=10,gpu=0,wall=15,turns=40`. No joint comparison is added to S0.
- GSM8K, Gemma3-4b-pt revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`,
  CLI 2.1.219, task assets, containers, evaluator and judges match S. PTB e62036f
  contains only the already recorded BFCL/HumanEval asset additions relative
  to the earliest GSM8K raw/protocol receipts; S itself already uses e62036f.
- Preserve input isolation and semantic audit requirements for both policies.
  The historical policy's old-runtime flags are not cleared by this comparison.
  No WMA schema, scorer, guards or common runtime is changed.

## Readout, falsification and costs

Primary descriptive comparison: S minus S0 final official accuracy, with all
four attempts, completion/judge/manual validity, individual scores, mean and
sample SD shown. Compare within GSM8K only. Also report original L0/L1 failure
recall where opportunities exist, L2 coverage alongside width/noise, actual
pre/post-advice actions and correct request/version timing. Distinguish final
verdicts from immutable request archives; do not double-count archival copies.
No outcome-derived card labels or new rejection quota are introduced.

Failure to observe better scores/actions with preserved validity and costs
leaves a policy improvement unestablished. A higher maximum alone, broader L2
intervals, zero-denominator recall, or unobserved counterfactual savings cannot
establish improvement. This tests the frozen policy bundle, not a single line.

Guards remain zero scope/leak flags including separate semantic audit; accepted
review cost no more than 1.5x the S0 reference; PTB not below the reference beyond
its observed spread; no unjustified killed work; no run before its blocking
review returns. Record all attempts, failures, timeouts, waits, total available
scientist/WMA/judge spend and allocated GPU time, with missing costs explicit.
Existing raw early-exit failures are not replaced or used as these arms' controls.

Four repeats are exploratory. Formal replication and held-out promotion gates
remain unchanged, including AIME promotion-only. Scheduling is not randomized;
report start-time differences and do not claim replicate IDs imply common
random numbers or a paired design.

## Launch gates and continuation

1. Validate the archived policy's schema/example and existing skill contract;
   compare every shipped non-skill private file with `31b854bb`.
2. Register one validation-only context/runtime smoke cell v91r01, in a separate
   four-cell technical manifest. It uses the exact legacy private archive and
   common public archive, actual SIF/model/MCP and isolation acceptance. The
   other technical slots are not automatically launched. This produces no
   scientific score and is excluded from completion counts.
   Use a new S0-specific context-validation record path: a smoke must never
   overwrite the context proof already frozen into existing queued receipts.
3. Submit the technical job held, register the receipt, verify nodes 2–3, then
   release. Use the current subqueue; no cancellation or borrowed capacity.
4. After its successful artifact is available, add that exact acceptance path
   to all four S0 cells, run the full manifest/site gates, inspect reconcile
   preview, and submit the four scientific cells asynchronously. A pending or
   failed validation is not acceptance. No permission needs to be re-requested;
   these steps follow the user's existing experiment authorization.

Added budget: **40 nominal scientific GPU-hours**, plus preparation/grading,
and at most **0.5 allocated GPU-hour** for the one technical smoke's wall cap.
Do not fill capacity with extra repeats or turn this into a runtime-fix study.
