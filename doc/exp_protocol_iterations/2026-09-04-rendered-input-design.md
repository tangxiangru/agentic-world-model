# Rendered-input component: planner design decision

2026-09-04. E7 construction specification, not a new experiment or a GPU result.
The focused reviewer read original g01r03/g01r05/g01r08 scripts and raw traces;
the planner independently reread the anchors below and rehashed all three traces.
No training, evaluation or tokenizer command ran for this review. This record
is the planner's adjudication, not an unaltered transcript of the reviewer's reply.

## Evidence that changes the interface

| Case | Direct source | Consequence |
|---|---|---|
| g01r05 |trace3428–3435,3530–3549: renderer appends EOT plus newline; labels supervise the full suffix, before raw0/500 failure at4898|Raw suffix was not actual training termination. Requiring the final label token alone to be EOT would also misclassify its valid newline tail.|
| g01r08 |trace3487–3503 appends EOT in trainer;4149 raw0/500 failure;4174–4181 moves EOT into builder and removes the trainer append|A second representation-move case. Retained post-repair scripts alone conceal what the original execution would consume.|
| g01r03 |retained trainer33–58,trace3810–3835: already-rendered prompts and completions tokenized separately, then concatenated and masked|Do not apply another chat template or require joint-tokenization equality for this separate-concat recipe.|
| g01r08 |retained collator82–90 pads batch width to a multiple of8 and masks padding|Padding width is not retained sequence length or truncation.|

Trace SHA256 (`solve_parsed.txt.gz` under guard-x8-v1):

- g01r03: `44ecdfc7346a6491e0c1f59ce425d88368d90e3c8f84beb93cbfc925e5c3d553`
- g01r05: `8ab1fcc25c99aa636a600d4da9211b0089c3b68b0da2ef150603637b8dff2fc9`
- g01r08: `a9bc4fdd35635f89fac7f209a29e29b3e4f3c8ff658f315a68fac43e23a36625`

The Window04 [semantic audit](trace-reviews/window04-local/card-semantics-audit.md)
remains valid. This adds the independently checked g01r08 counterexample; it does
not claim that either schema move caused a score gain or rescued a failed run.

## Accepted design: one prepared token artifact, one checked consumer

Use an opt-in `RenderedTrainingBundle` for unpacked, completion-only causal-LM
training. The materialized token arrays checked before launch must be the arrays
loaded by the supported training loader/collator. A separate diagnostic rerender
and an asserted summary JSON cannot certify the unchanged training pipeline.

```python
# CPU preparation before lock; no model construction/forward.
bundle = RenderedTrainingBundle.prepare(
    sources=[raw_jsonl], render=shared_render_function,
    tokenizer=local_tokenizer, template_bytes=actual_template,
    settings=explicit_settings,
    source_files=[training_script, preprocessing_module, renderer_module],
    output=token_bundle_directory,
)
# Put receipt path/hash in the card, then check/lock.
# Inside the locked command:
bundle = RenderedTrainingBundle.open_for_training(card_path)
dataset = bundle.dataset
collator = bundle.collator(pad_to_multiple_of=8)
```

Own tokenization, supported masking, filtering and padding after the renderer
returns strings. Support `separate_concat` (prefix+target separately tokenized)
and `joint_prefix` (prefix+full text tokenized, exact prefix-token alignment
required). Do not silently substitute one for the other. Bind the explicit
template snapshot actually consumed by the renderer; a filename hash cannot
prove a cached compiled template used the same bytes.

Use one kept token record per source occurrence with `input_ids`, `labels`,
`target_start`, source hash/row ordinal and explicit keep/drop reason accounting.
The card's named training data must reflect what the loader really reads.
Prefer reusing materialized storage over a redundant diagnostic copy. Measure
preparation/check/loader cost on representative CPU data before freezing E.

## Mechanical validation boundary

- Structural invariants are hard: integer IDs, matching lengths, nonempty
  supervision, prefix labels ignored, unshifted target labels equal input IDs,
  and no supervision on padding. Custom shifted/multi-span/packed labels are
  unsupported, not silently accepted.
- Resolve the stop sequence from the effective tokenizer. Validate the
  supervised terminal sequence plus explicitly bound template tail; test
  newline tails, non-106 IDs and supported multi-token stops. Do not inspect
  only raw strings, the last/penultimate position, or masked prompt tokens.
- Count answer markers within the decoded supervised answer span only, with
  cleanup settings bound, excluding prompt instructions/demonstrations and
  template tail. Keep semantic tolerance policy explicit: current raw defaults
  are95% stop consistency and at most2% bad markers, not a new universal100%
  criterion silently introduced by the helper. Report actual rates.
- Initially support drop-overlength, not hidden truncation. Retained nonpadding
  length must fit; padding-to-eight alone is not an overlength example.
  Truncation requires a future explicit pre/post supervision-preservation adapter.
- Reconcile source rows, considered rows, excluded-by-declared-limit, kept rows,
  overlength drops, prefix-drift drops and other supported reasons. Malformed
  rows/filtering cannot disappear from the denominator. Report pre/post length
  distributions and rejected-row locator digests. A high drop rate is visible;
  its scientific acceptability is not inferred from valid remaining tokens.

## Receipt and optional v2 integration

```yaml
setup:
  rendered_training:
    receipt: memory/preprocessing/exp-02.receipt.json
    sha256: <receipt-bytes-hash>
```

Receipt identity must cover adapter/code; all raw inputs and actual token
artifacts with hashes/counts; producer scripts/modules; effective tokenizer
state and local assets; explicit template bytes; tokenization/masking/filter/
limit/seed/padding/renderer settings; and mechanically derived coverage/findings.
Include added tokens, special IDs, normalization/postprocessing and active
padding/truncation settings in tokenizer identity. Unknown/custom tokenizer
state requires an adapter, not a partial fingerprint labelled complete.

Preflight rechecks actual artifacts and recomputes token findings; it must not
execute arbitrary renderer code named by a card. Cache reuse depends on content
and settings, not mtime/size. Avoid a receipt/plan hash cycle: preparation does
not contain the final plan hash; the training loader subsequently verifies and
binds the receipt to the matching successful lock. Source changes invalidate it.

Valid complete supported token evidence makes raw suffix/marker/length findings
advisory/superseded, not a relabelled raw PASS. Stale/malformed claimed evidence
fails its own check. Without opt-in evidence, rendered coverage stays unverified
and the existing raw/recorded-override path remains; a claim that a renderer
fixes the input is not enough. No required v2 field is added; a future mandatory
format requires an explicit v3 migration.

## Required combined tests and claim limits

Reproduce both old/new g05 and g08 representations with identical supervised
tokens; exercise g03 without double rendering; keep tokenization modes distinct.
Include masked-only EOT, masked terminal labels, duplicate terminators, template
tails, prompt-only markers, prefix drift, all-ignored labels, double shifting,
template-added overlength, batch padding, declared limits/drop accounting,
changed sources/settings/tokenizer/template/artifacts and aggregate-only forgery.
The loader rejects absent/stale locks; supported collator rejects changed arrays
and transformations inconsistent with the prepared bundle. No model is needed.

Report **verified preparation**, **observed loader/collator consumption**, and
**unknown actual model consumption** separately. Arbitrary scientist code can
bypass the loader or transform a later batch. Neither preparation nor consumer
records prove optimizer execution, strategy quality, contamination absence or
serving-distribution equivalence. The planner must still audit actual launches
and consumers in scientist traces. No new validator-clean cells are counted.
