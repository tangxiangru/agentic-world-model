# Round 01 online — 2026-09-02 — skill v0.2 against no-WMA control

**Status:** running; this is an open record. No promotion or skill change is
allowed until the validator-complete comparison boundary below is met.

## Variants

| label | private WMA commit / skill hash | what differs from control |
|---|---|---|
| `wma-v0.2` | first wave `e8a8599`; v2 runtime `bf87dfb`; v3 runtime `34535c7`; skill `176f0a464986` throughout | private Claude Opus 5 / high online sidecar, train-only history and `cpu=10,gpu=0,wall=15,turns=40` |
| `ctl` | no private WMA checkout | same public checkout and protocol; the only difference is no `wma` manifest block |

`bf87dfb` is a registered measurement/runtime repair, not a skill variant: it
allows the schema's legal pre-launch `result.execution: not_run` sentinel.
`skills/wma/`, the prompt, verdict schema, scorer and history are unchanged.

## Cells

Task: `gsm8k` · base model:
`google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d` ·
scientist: `claude-opus-5[1m]`, high, 1M context · 10 h · one H100 per cell.

- Core: first-wave `w01r01..08` and `c01r01..08` (8 per arm).
- Pre-result precision extensions: fixed-runtime v2 `w02/c02` and `w03/c03`
  (16 more per arm). The v1 buffers were cancelled while all 32 jobs were
  PENDING; they never replace a result.
- Independent probe-basis-runtime extensions: v3 `w04/c04` and `w05/c05`
  (16 more per arm). They are fresh paired observations, not replacements for
  v2 cells that had already started.
- Held-out task this round: none. `aime2025` remains promotion-only.
- Primary end-to-end view: WMA **as attached** versus control. Sensitivity:
  **as answered**, requiring at least one valid verdict in that WMA cell.

## Results

No cell is terminal or validator-complete as of 2026-09-02 19:21 UTC. The WMA
subqueue's owned nodes remain 16/16 allocated. First-wave 16 and 27 v2 cells
are RUNNING; five v2 cells plus all 32 v3 cells are PENDING, restoring the
required waiting waterline to 37. Therefore there is no PTB score comparison
or final online ledger conclusion yet. The closed-card observations below are
provisional training-side evidence exposed to later, pre-launch WMA reviews;
they are not terminal PTB results.

### Inflight WMA evidence

- All eight WMA scientists invoked `exp_protocol` as their first tool action
  and all eight sidecars started with the frozen Opus 5 / high contract.
- Seven cells have now produced 26 review transcripts: `w01r01` (5),
  `w01r02` (3), `w01r03` (3), `w01r04` (4), `w01r05` (3), `w01r07` (5), and
  `w01r08` (3). The first verdicts used 28–32 of 40 turns, 5.1–7.2 min of 15,
  and $1.46–$1.72 shadow cost. Isolation stayed within `/session`, train-side
  `/history`, private skill and scratch.
- The old runtime accepted 21/26 payloads and moved five to `.rejected`:
  `w01r01/exp-03`, `w01r02/exp-04`, `w01r04/exp-04`, `w01r07/exp-05`, and
  `w01r08/exp-04`. All five cited a recorded `probes[].id` from at least one
  level's `basis`, while the validator accepted only `evidence[].id`; two also
  carry a leak flag and remain excluded from a clean ledger independently.
  The schema relaxation in this checkpoint accepts evidence or probe ids and
  leaves the prediction untouched. Replaying all 26 transcript payloads under
  it validates 26/26. The provisional 12-card ledger below uses only verdicts
  that the frozen runtime accepted, so its denominator does not change.
- `w01r03`, `w01r05`, `w01r06` pre-filled `result.execution: not_run`; the old
  runtime rejected exp-01 as post-hoc. This is infrastructure evidence, not a
  WMA miss. It caused `bf87dfb` and the v2 buffer replacement. `w01r03` and
  `w01r05` recovered and began receiving verdicts at exp-03; `w01r06` has not
  produced a verdict. The rejected exp-01/02 requests remain in the as-attached
  cohort and out of as-answered.

### First four known card outcomes (not terminal PTB results)

The exp-02 transcripts expose the already-closed exp-01 proposal cards for
`w01r01`, `w01r02`, `w01r04`, `w01r08`. Their dev-150 base-checkpoint
accuracies are `{0.0667, 0.0533, 0.0533, 0.0333}`. All four found the same
mechanism: the pretrained checkpoint often fails to emit `<end_of_turn>`,
continues into fabricated Q&A, and the end-anchored numeric grader reads a
number from the tail. None shows the feared `!!!!` corrupted-serving signature
at `--max-connections 16`.

- L0 `yes` is 4/4 correct.
- L1 `yes` is 4/4 correct; the invocation ran validly despite the stated
  concurrency risk.
- The WMA named the termination mechanism correctly but put too little mass on
  the near-zero mode.
- The pre-registered Window 02 absolute-accuracy-slot candidate is falsified:
  recorded absolute intervals were `[0.02,0.40]`, `[0.10,0.50]`,
  `[0.12,0.48]`; two of three miss low. The missing slot is not the current
  bottleneck—the prior is.

All four exp-02 plans independently converge on completion-only SFT over
GSM8K-derived OpenMathInstruct-2 data with `<end_of_turn>` targets. Their WMA
L2 delta intervals are 0.30–0.40 wide, about 10× the n=150 noise floor. Whether
that width is earned remains open until the measured exp-02 deltas arrive.

### Provisional closed-card ledger

Later review transcripts expose the frozen result sections of earlier cards.
Among cards with both a valid verdict and an observed comparator/result pair,
the provisional L2 coverage is:

| card family | scorable | in interval | miss side |
|---|---:|---:|---|
| exp-02 completion SFT | 5 | 3 | 2 above |
| exp-03 decode/config | 3 | 2 | 1 below |
| exp-04 follow-on SFT | 4 | 2 | 2 above |
| **total** | **12** | **7 (58%)** | **4 above, 1 below** |

Fable commit `82cf601` makes the corresponding read-time scorer distinction
explicit. A first card that measures its own previously unmeasured base
comparator is valid at L1 when it produces a reading, but its self-delta and
next-step decision are unscorable at L2/L3. Across all 15 accepted closed
cards this corrects L1 from 12/15 to 14/15 and removes three self-measurement
cards from L2, yielding the same 7/12 table above. The intervention-only 12
cards retain L0/L1 11/12.

The same 12 cards give provisional L0 and L1 hit rates of 11/12. The sole
shared miss is `w01r07/exp-02`: WMA predicted that the launch would not run or
produce a valid candidate, but it completed, scored 0.6533, and was adopted.
An audit after Fable Window 04 confirms this is a real predictive miss, not a
heeded warning. The protocol intentionally launched asynchronously at
14:20:10 and requested review in the same command; the verdict landed at
14:28:59. The locked launcher hash still matches, `train_sft.py` predates the
lock and still sets `use_liger_kernel=True`, and no precondition could have
changed the already-running launch. WMA correctly found that Liger was absent
but incorrectly inferred a fatal ImportError; this Transformers path simply
skipped the unavailable kernel and trained. The lock only hashes the launcher,
not imported helper scripts, which is a separate measurement weakness to
track. These rates must be recomputed by `awm wma ledger` after collection;
the table is an inflight audit, not a substitute for the frozen truth files.

The miss direction matters for candidate selection. Four of five L2 misses
are real improvements above the predicted upper bound. Even the existing
0.30–0.40-wide exp-02 intervals cover only 3/5 outcomes. The competing proposal
to cap interval width at a multiple of the n=150 noise floor is therefore
falsified for this round; it would worsen coverage rather than calibration.

All eight WMA-arm and all eight control-arm exp-01 cards were read directly
from their live receipt-backed cells and recorded with complete-card SHA-256
locators. Their 17 measurements (one card has stock and greedy arms) all fall
in 0.0333–0.0867; 15 use n=150 and one control uses n=200. Every card names
the same cap-hit / missing-termination / trailing-Q&A mechanism. This passes
the first cross-arm falsification check for a termination-floor prior on the
complete first wave, not just its visible half. It also shows that `<= 0.07`
must not be a hard upper cutoff: the useful prior is a low-score band of
roughly 0.03–0.09, centred near or below 0.07, coupled to the diagnostic. The
second check is resolved below.

### V2 pre-change falsification check

All 16 v2 WMA cells (`w02` and `w03`) produced an exp-01 verdict under the
unchanged v0.2 skill. Ten correctly treat the baseline card's structured L2 as
a delta against itself and centre it near zero. Six instead put an
absolute-looking interval in the same delta field, with upper bounds 0.30–0.45;
some simultaneously label the direction `flat`. None gives a consistent
structured 0.03–0.09 absolute band. Thus the second falsification condition
does not fire: v0.2 does not already encode the format-floor prior reliably.

This also corrects the promotion readout proposed in Fable Window 04. The
current ledger scores L2 delta, so an "exp-01 in-band rate" is not mechanically
defined without adding a second schema field. Round 02 should remain a
single-policy change and evaluate the prior through first-format-fix SFT L2
coverage, diagnostic use, and manual exp-01 absolute-estimate audit. Adding an
absolute prediction slot in the same round would confound the policy change.

## Decision

Inconclusive. Runtime/isolation/procedure are supported for seven answered
cells. The inflight ledger rejects the interval-width cap and strengthens the
termination-floor candidate, but skill calibration and PTB effect do not yet
meet the terminal evidence boundary. Keep skill v0.2 unchanged.

The end-to-end conclusion requires at least eight validator-complete cells per
arm. A skill candidate additionally requires closed-card ledger evidence and
the pre-registered falsification checks below.

## Change

None to `skills/wma/`.

Measurement-only changes already made and separately attributable:

- `bf87dfb`: accept `not_run` as pre-launch in the post-hoc guard.
- `2b094c7`: replace only unstarted v1 buffer cells with paired v2 manifests.
- `34535c7`: allow a level's `basis` to cite its own recorded probe
  ids as well as evidence ids; 53 focused schema/backend tests pass and all 26
  inflight payloads validate. No scorer, prediction, skill file, or skill hash
  changes.
- `3b2d4f1`: verify every held formal job's scheduler `ReqNodeList` before
  release. Existing v2 receipts recorded the right subqueue but some running
  jobs now show `ReqNodeList=(null)` outside the four-node scope; those RUNNING
  jobs were not cancelled. All 32 v3 jobs passed both held and post-release
  checks and remain PENDING on `slurm2-a3nodesetondem-[2-3]`.
- `82cf601` (Fable): score a first self-measurement on whether its reading
  exists at L1, and leave its self-delta and next-step decision unscorable at
  L2/L3. Decode comparisons with a real comparator path remain scorable. This
  reconciles the mechanical ledger with the pre-registered 12-card audit and
  changes no verdict or skill bytes.
- `2b6bfeb` plus retry receipt: cancel only unsafe legacy v2 PENDING jobs
  `90633..90637` (`c03r04..08`) after confirming each was still PENDING, then
  requeue the same five cells as `90786..90790`. Every retry passed the held
  route gate and remains PENDING with
  `ReqNodeList=slurm2-a3nodesetondem-[2-3]`; 32 safe v3 jobs stayed queued, so
  the operation never approached the eight-job floor.

### Terminal disposition for out-of-scope v2 jobs

The 27 already-running v2 jobs with `ReqNodeList=(null)` are never cancelled
by this round. Their terminal state is handled per cell, not by blanket policy:

1. Resolve receipt → cell → manifest → spec → result directory, harvest with
   `--all`, and run the PTB validator. Slurm completion alone is not a result.
2. A validator-complete result is preserved and analysed *as run* with its
   actual node provenance. Placement noncompliance is reported explicitly and
   the cell is kept separate from the canonical owned-node view until hardware
   and runtime equivalence are audited; it is never silently substituted.
3. A failed/killed/incomplete result caused by infrastructure or routing is not
   a scientific negative. Exclude it from the primary paired denominator.
4. Requeue the exact cell through the held-route gate only when its missing
   arm is still needed for the pre-registered pair/minimum-n decision. If an
   independently specified safe extension already supplies adequate evidence,
   record the failure without duplicating it. Retry and original are never
   counted twice.

Monitoring is now every 30 minutes. The hard requirement is at least eight
safe PENDING jobs; the operational guard fires below 24 and replenishes to at
least 32. With at most 16 owned GPUs starting one wave between checks, this
keeps an eight-job reserve without high-frequency polling.

The repeats-4 redesign must preserve that invariant during the transition.
The 32 routed v3 jobs remain submitted safety inventory until all Round 02
replacement manifests are frozen and submit-ready. Cancellation and replacement
are one previewed cutover leaving at least 24 safe PENDING jobs; the operator
must not apply the proposed intermediate state of 13. If an inventory job starts
first, it finishes naturally and is analysed in its actual cohort.

## Evidence

- Receipts and inflight snapshots under
  `results/ptb/wma-gsm8k-gemma4b-high-r01-*`.
- Complete first-wave base-check provenance:
  `doc/wma_iterations/evidence/2026-09-02-round-01-base-checks.md`.
- PR 23 Fable windows:
  [Window 02](https://github.com/tangxiangru/agentic-world-model/pull/23#issuecomment-5510983693),
  [Window 03](https://github.com/tangxiangru/agentic-world-model/pull/23#issuecomment-5511204993).
- Runtime diagnosis and replacement contract:
  `doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`.

## Next round

Primary proposed single change, not yet applied: add a C11/C12 manual prior
that a pretrained non-instruct checkpoint evaluated through a chat template
with end-anchored numeric grading should place its expected result in the
termination-floor band (currently about 0.03–0.09, centred `<= 0.07`), and
demand the cap-hit / missing-answer diagnostic.

Pre-registered falsification:

1. If matched control exp-01 values are not also at the floor, the observation
   is WMA-arm-specific and the prior is wrong.
2. If the v2 pre-change WMA exp-01 verdicts already centre on the floor, the
   entry adds no useful discipline.

Check 1 passes on all 16 first-wave cells. Check 2 also passes in the sense
required for a non-redundant candidate: the 16 v2 verdicts are inconsistent
between zero-delta and broad absolute-like interpretations and never express a
stable floor band. Promotion must use first-SFT L2 coverage rather than a
nonexistent structured exp-01 absolute field.

The competing interval-width cap is now rejected by the provisional ledger:
the wide exp-02 intervals were needed and still missed high twice. Re-open it
only if the frozen ledger contradicts the 12-card inflight audit.
