# L: materialized data and numerical observations

2026-09-04. Construction contract for the complete E+L package, not a new
manifest or model experiment. The reference is the identical completed E core
in [bundle discovery](../spec/2026-09-04-exp-protocol-bundle-discovery.md).
Implement on a separate E+L branch after freezing E; do not silently add L to
the E reference or change the operator's shipped runtime.

## Evidence and intended decision

The opportunity review's `high-case/audit_data.py` read the actual g01r03
JSONL and tokenized it: 253,045 and350,000 rows had70,315 and69,221 exact
questions, but only93,118 common exact question/solution pairs. More rows did
not mean more questions or a superset. Its format-specific prefix parser
verified actual text against `nshot`; that parser is not a universal prompt
grammar. `counterexamples/audit.py` reports **prefix metadata** separately,
not proof that those prefixes were rendered. The g01r05 builder's repeated
dictionary references and later in-place edits changed the materialized prefix
distribution. These are reasons to inspect actual occurrences, not prescribe
350k rows,8% few-shot or a particular recipe.

The g01r04/08 scripts used BF16 parameters directly; g01r03/05 used FP32
parameters with BF16 autocast. Retained pinned optimizer source distinguishes
moment/state storage from actual parameter updates. There is no historical
measurement of weight deltas or evidence assigning the score gap to rounding.
L must collect that missing observation without declaring flat loss to be
saturation or treating FP32 optimizer state as FP32 master parameters.

Sources: operator checkout's
`doc/exp_protocol_iterations/analysis-2026-09-04-opportunity-review/`
(`high-case.md`, `high-case/audit_data.py`, `counterexamples.md`,
`counterexamples/audit.py`, and the retained optimizer-source provenance).
The analysis drafts are owned by the concurrent user task; do not stage them
as part of L construction.

## 1. Census the actual materialization

Provide a CPU-only data-observation helper plus a small optional E7 integration.
The observation point is the **same call** that produces actual rendered
prefix/target/full text and the actual kept/drop decision. Do not invoke the
renderer a second time: randomness, aliases and mutable state could differ.
When no L observer is requested, E7 behavior and evidence claims stay unchanged.

Freeze raw field-view values/digests before calling the renderer: E7 currently
passes it a mutable decoded row, so a later view could observe in-place renderer
changes and mislabel them as raw input. Preserve the renderer's existing input
behavior; emit a separate immutable rendered/decision event afterward.

The observer receives source index/line/raw-byte hash, the decoded materialized
row, actual rendered strings, token/label lengths and kept/drop reason. Rows
excluded by a declared global limit remain separately counted, with no invented
rendered observations. Bind the final observation receipt to the completed E7
receipt/hash, raw sources, renderer/adapter source, settings, token file and
decision ledger. Failed preparations retain partial observations, never a
complete-census claim. Verified reuse must either reuse a matching complete L
receipt or explicitly report missing L coverage; it must not simulate a fresh
observation of an old renderer execution.

Finalize L only after E7's final whole-bundle verification succeeds; E7 writes
`receipt.json` before that check, so file existence is not completion. Keep the
final reference one-way, L to E7, without an E7/L hash cycle. Prefix-drift drops
have no valid supervised-label span; their token observations must not fabricate
supervised counts. Limit-excluded rows have raw observations only.

Use explicit source-backed views for question, solution and prefix structure.
Ordinary field views extract exact strings from named materialized row fields;
format-specific views can identify spans in the actual rendered strings. Bind
the chosen view implementation and its settings. Missing/ambiguous structure
is unknown or unsupported, not zero prefixes or an empty solution. Declared
`nshot`, `source`, desired proportions and caps are separate metadata. A view's
semantic interpretation is an explicit adapter boundary, not something a
hash or generic JSON parser independently proves.

For all raw occurrences, considered occurrences and kept occurrences separately,
report exact row count, unique question count, unique question/solution-pair
count, duplicate multiplicity and question/solution overlap fingerprints.
Use unambiguous typed/length-framed canonical encodings, not delimiter joins
that collide on embedded NULs. Do not collapse duplicate rows before counting
their training weight; do not normalize whitespace/case without an explicit
different profile. Report actual prefix distribution and unknown coverage,
nonpadding/supervised token totals and length distribution after filtering.
Question/solution counts must distinguish raw views from rendered views when
they differ. Token counts describe prepared inputs, not proven optimizer use.

Retain per-occurrence Q/QS digests with locator/decision and unique-digest
multiplicities so P can compute overlap counts, not just compare aggregate
fingerprints. If some views are unknown, the full-population unique count is
unknown too; report the known-subset count and coverage separately.

Keep compact per-occurrence source/render/token bindings and derived hashes,
not another kept-token cache or an automatic copy of every training string.
Verification reconciles counts and hashes against materialized evidence and
labels derived adapter observations honestly; it must not turn an arbitrary
scientist-authored summary into independent semantic validation. Record CPU
time and disk footprint on varied data including duplicates and long rows.

## 2. Observe numerical state and bounded parameter changes

Provide an opt-in observer inside the existing locked training command. It
does not construct a model, run a forward, choose an optimizer, perform extra
steps or change the precision/LR/recipe. CPU construction tests may use tiny
native tensors and optimizer updates, never a benchmark training run.

Capture separately:

- Actual named parameter dtype/device/count at the observation point, frozen
  and trainable coverage, and native optimizer class/source identity.
- Actual allocated optimizer-state tensor dtypes/counts and missing/lazy state.
  Master-parameter storage is unknown unless a specifically supported adapter
  observes it; FP32 moments must never be labelled FP32 master weights.
- Autocast state/device/dtype at the observed point, distinct from a caller's
  declared forward precision. A probe outside a forward context cannot prove
  how prior forwards ran. Export dtype is separate artifact evidence, not an
  inference from live parameters or a filename.
- Deterministically selected parameter names and element offsets before/after
  one caller-controlled update boundary. Declare the maximum tensor/element
  budget, selected strata and missing coverage. Compute deltas/norms in a
  sufficiently precise CPU representation; report observed zero-change fraction,
  absolute/relative sampled delta and finite/nonfinite counts. A sampled norm
  is not a full-model norm, and zero changes do not identify their cause.

Prefer observation around the actual supported optimizer-step call. Record
whether that call was observed once, skipped, failed or unsupported; two arbitrary
snapshots must be labelled an interval, not a verified optimizer update. Preserve
the original optimizer return/exception and do not retry or run another step.
Keep `step_call_status` distinct from `observer_status`. No observed step call
does not prove a GradScaler skip unless a supported scaler adapter observes
that condition. A returned step does not by itself prove nonzero updates.
Record synchronization/copy/observer cost separately from step wall time.
Gather only the selected bounded offsets on their device before CPU conversion;
do not clone/flatten/cast an entire large tensor to obtain a small sample.
Changing parameter identity, shape/device, concurrent access, custom/distributed
optimizers or unsupported state must not produce a fabricated normal result.
Support ordinary single-process native Torch updates first and explicitly
inspect the pinned bnb path needed by this scientist environment; unsupported
adapters remain visible, never an automatic recipe rejection.

Persist unique observation records with live card/plan/lock/source bindings,
process identity, timestamp and step/interval identity. Do not require a new
card for a measurement inside its already-covered training invocation, waive
the original lock, or make model-free preprocessing look like a model probe.
No automatic halt or remedial recipe change follows a low-update observation.

## 3. Reduce duplicate entry without erasing the experiment card

Expose a compact, hash-bound evidence reference and a human-readable summary
for the card's existing record/interpretation fields. Derive actual counts and
measurement facts from those records instead of asking the scientist to copy
them into multiple tables. Keep hypotheses, decisions and unknowns authored
explicitly; do not auto-fill predictions, causal explanations or acceptance.
Preserve required v2 fields and card/check/lock ordering. Any optional structured
reference must be validated by its real consumer, or stay an ordinary cited
record rather than an unused schema field.

## 4. Real interaction with P

L exports versioned census and numerical-record references with path/hash and
source/E7/card identity. P can attach these to an explicitly named retained
checkpoint lineage, revalidate available references, and compare observed
question/solution overlap and update coverage alongside paired outcomes.
Missing lineage or stale references remain unknown. A training-data difference
does not establish why a paired score changed. P never imports test-derived
IDs, questions, targets or error examples into L training observations.

E+L+P requires an actual tested consumer of these references, not merely both
guides installed. Test at least two independently materialized training views
linked to two retained artifacts and a measured combination. The diagnostic
report must distinguish same questions/different solutions, artifact retention,
matched scoring and the absence of any causal or oracle-deployment claim.

## 5. Acceptance and negative cases

Cover repeated aliases serialized after mutation; same question/new solution;
delimiter/typed-identity collisions; raw versus actually rendered prefixes;
malformed/unknown views; duplicates split between kept/dropped rows; excluded
limits; stale sources/receipts; failed materialization and reuse without L
coverage. Reconcile every occurrence and token count with real E7 artifacts.

Numerical tests cover native FP32 and BF16 parameters, FP32 state without
master weights, lazy state, frozen parameters, deterministic sample coverage,
true zero/nonzero changes, nonfinite values, parameter replacement, exceptions,
unsupported optimizer/serialization cases and observer overhead. No model
quality, full-model saturation or GPU-throughput assertion comes from them.

Run the complete E regression plus L tests and an independent public-guide
forward pass. Freeze a full E+L identity only after these interfaces work,
including reduced duplicate recording. All components above are part of L;
passing a raw-row counter alone is not a smaller substitute package. The
two-cell discovery budget, useful-held floor and native release gates remain
unchanged.
