# Exp-protocol bundle discovery — 2026-09-04

Status: construction specification; **not a manifest, submission, or release**.
The [user-directed search policy](../../skills/exp_protocol_meta/search_policy.md)
supersedes the saved goal's historical single-item wording. No additional
strategy confirmation is required. Native isolation/release authority is a
separate, still-unmet gate. Historical standalone candidates are components,
not automatically funded GPU arms.

## 1. Evidence and comparison contract

Reference scientist tree: guard drift `2f64581`, protocol tree
`189319d63d301d64d96f8f41d051795404679f37`. Freeze every complete package as a
new commit plus protocol tree and the exact six `EXP_PROTOCOL_SHIP` paths.
Do not change the host runtime tree while constructing isolated candidates.
Keep PTB `dcf5da031435c54e3680b6ec3f63e7e317efc13e`, GSM8K,
`google/gemma-3-4b-pt`, the existing scientist/model/high-effort/1M context,
10-hour budget, images, resources and official evaluator contract matched.
The analyst's max effort is not the scientist's effort setting.

Sources are the closed [Window04 decision](../exp_protocol_iterations/2026-09-04-round-02-window04-decision.md),
its retained raw cards/traces and focused audits, and the
[opportunity review](../exp_protocol_iterations/analysis-2026-09-04-opportunity-review/README.md).
Corrections remain authoritative over the original helper narratives: g01r03
really used about8% few-shot-prefixed rows; stored sample-array order is not
dataset-prefix order; unchanged logs/low GPU memory are not exit timestamps;
g01r05's original renderer already supervised EOT. None of these observations
isolates a package's effect or changes the official82.79% record.

This is an **outer protocol comparison**, not fixed-recipe replication.
Scientist choices of data, precision, optimizer and strategy are outcomes.
No package prescribes350k rows,8% prefixes, FP32, RFT, soup, greedy decoding,
or a target score as a universally superior recipe. No benchmark test IDs,
questions or gold answers enter training, watch sets or failure examples.

## 2. Packages and limited spend

| Package | Complete treatment | Decision value | Discovery |
|---|---|---|---|
| E | Integrated execution/recording core below | Do interacting reliability repairs yield more usable research without false blocks? |2 independent cells|
| E+L | E plus actual-data/numerical observation and reduced duplicate entry | Does automatic materialized evidence reduce erroneous records and diagnostic cost? |2|
| E+P | E plus bounded branch inventory and matched-item diagnostics | Does retained evidence expose useful branches and measured combinations beyond scalar ranking? |2|
| E+L+P | Both additions on the identical E core | Do automatic observations feed branch decisions usefully, or does combined complexity negate their value? |2, only after the interaction is implemented/tested|

The fourth package has a concrete interaction question, but is not filler:
if L/P cannot actually consume each other's evidence, defer it. A completed
declared6-cell discovery wave may trigger review. No package is frozen yet.
Default at most2 further cells per configuration only for a written unresolved
decision. Count all attempts across aliases and strict-site replacements.
No automatic new null controls, drift pairs, or winner-to-eight expansion.
Reuse existing reference evidence and disclose historical site/time differences.

## 3. E component ledger and joint interfaces

All listed components are required before calling E construction complete.
Partial implementation is a component prototype, not a smaller substitute arm.

| Component | Evidence / reusable candidate | Required joint behavior |
|---|---|---|
| E1 launch scope | J `549e25a`; Window04 launch audits | Every training/evaluation command, including smokes and probes, has a prior matching card/check/lock. CPU-only preparation is distinct. Retrospective smoke entries never confer coverage. |
| E2 honest data applicability | H `b52e5f2`; fictitious eval-only data entries | Non-training cards need no invented training data. A family label is not proof that the command does not train; audit actual operation, and preserve required data for real target-text training. |
| E3 comparator lifecycle | K `58a6992`; mismatched/guessed comparator records | Preserve optional deferred-comparator mode, immutable plan, real completed count/metric checks, failure closure, portable receipt, index/collect/Stop-hook consumers. Preserve legacy raw counts separately. |
| E4 actual save safety | Withdrawn D scope audit; g03 merge, p01 Trainer and repaired p02/g05/g06 scripts | Check effective in-memory serialization state at actual supported save boundaries; protect both Trainer checkpoints and direct/merge saves. Never globally block eval-only cards or reject a repaired model from parent JSON alone. See section4. |
| E5 process and artifact completion | Revised E2 `c6f11d8`, with its unsupported strict-exit source claims corrected | Retain producer identity and exit result; verify outputs from the current invocation. Do not infer success/death from stale files, quiet tails, GPU memory or a missing PID. No long wait after known completion. |
| E6 sampling/serving boundary | B `9f294c3`, P5 serving audit, g08 diagnostics | Explicit effective prompt/tokenization/stop/parser checks; preserve raw samples before fallible parsing. Resolve stop IDs from the actual tokenizer/template, not universal hard-coded IDs. Scope engine cleanup to owned producer children. |
| E7 rendered training checks | Window04 g05 semantic audit | Validate the actual rendered/tokenized input and supervised labels, not raw-target suffix alone. Bind any preprocessing evidence to source/data/tokenizer/template; stale or unsupported evidence is not a PASS. |
| E8 export and available-tool guidance | g03/g05 save repairs; unavailable WMA CLI traces | Preserve selected on-disk serving settings through save/export, verify required processor/tokenizer files locally. Test installed WMA command availability before optional invocation; do not ship WMA/meta or wait for a reviewer. Remove the unsupported universal90%-execution attribution. |

E3's existing minimal-summary acceptance is not full dataset/model/decoder
identity verification. Do not extend its claim. An optional derived evaluator
summary must preserve source path/hash and actual Inspect completed/scored n;
requested limit, population size and SE inversion cannot supply missing counts.
Ordinary unsupported summaries receive an explicit unverified/advisory result,
not fabricated verification. P supplies stronger matched-item evidence.

For E5 use a thin foreground execution wrapper when practical, validating the
matching lock and declared command before execution and emitting a unique
attempt record with leader PID, start/end and exit result. It must propagate
failure and not silently chain dependent work. This is not a scheduler or an
LLM-gated state machine. A shell leader's exit does not prove internally
detached children finished; document the foreground-child contract and audit
bypasses. An optional wrapper cannot prevent arbitrary raw Bash launches.

E7 must exercise the same formatter used by training, including prompt masks,
terminal supervised tokens and truncation. CPU preprocessing before lock is
allowed; a model-forward evaluation is not. Missing adapter coverage stays
explicitly unknown. Do not turn a scientist-authored JSON declaration into a
claim that the training script executed those inputs. Any required schema
change needs a separately documented v3 migration; optional evidence may stayv2.

## 4. Save contract chosen for construction

Use a version-pinned, opt-in `GenerationSaveContract` and native Trainer save
adapter. The supported initial runtime is the frozen image's Transformers4.57.3,
ordinary single-process native `PreTrainedModel`/Trainer and direct native saves.
Unsupported remote/custom/FSDP/DeepSpeed/PEFT/TP/TPU paths must be explicit,
not silently certified. No model weights or training are needed for checks.

```python
saves = GenerationSaveContract(policy="inactive_sampling_v1")
saves.check_before_compute(model)  # after scientist in-code repairs
with saves.saving(model, output_dir, selected_serving_json=None):
    model.save_pretrained(output_dir)
```

Deep-copy both live config objects, project native migration from
`model.config` into `model.generation_config`, and invoke actual strict
validation. Native `modeling_utils.py:3915–3930` migrates immediately before
generation-config serialization: checking only the pre-migration generation
object is insufficient. If already valid, normalize nothing. Only for an
invalid greedy configuration may the serializer copy neutralize offending
sampling-only fields to pinned defaults (`temperature`, `top_k`, `top_p`,
`typical_p`, `min_p`, `epsilon_cutoff`, `eta_cutoff`); preserve `do_sample`,
special-token IDs, cache settings and other fields. Revalidate strictly;
remaining errors stop before writing. Do not classify by exception wording.

Recompute at every actual save; temporarily assign the validated copies and
restore original object references in `finally`, including interruption and
writer failure. Reject reentry/concurrent use; require exclusive model access.
Native Trainer integration surrounds the actual `_save` boundary, not
`on_save` (callback occurs after saving). Preserve native state-dict/tokenizer/
training-argument behavior and re-resolve replaced model objects.

A default output is a serializable checkpoint. An explicitly selected serving
export supplies frozen JSON bytes: after successful serialization install those
exact bytes atomically as `generation_config.json`, verify the hash, and record
checkpoint and serving identities separately. Restoring Python objects alone
does not restore what vLLM reads. A failed serving-file write makes the export
incomplete. Artifact consumers must verify the selected serving hash before
identifying an output as the selected artifact. The helper does not choose
decoding settings, guarantee whole-checkpoint atomicity, or prevent bypass.

Record input/effective hashes, migration and normalization diff, library
identity, output, outcome and serving hash without swallowing the original
exception. A plain effective-config validator without normalization is the
fallback design, not a reason to retain D's known false scope.

## 5. L and P boundaries

L: collect actual materialized row/unique-question/exact-solution-pair counts,
prefix distribution, post-filter token counts and source hashes. Separate
parameter/master/state/autocast/export dtypes. Optional bounded update-health
observations report measurement time, parameter sample and missing coverage;
flat loss alone is not saturation, and zero update is not automatically a
recipe rejection. Reduce duplicate manual entry by referencing these records,
not by inventing unknown facts or removing card ordering.

P: maintain an explicitly budgeted checkpoint inventory with lineage, size,
retention promises, actual existence and scientist-selected future use. Tools
report evidence; they do not automatically rank research directions or delete
promised/unowned artifacts. Pair structured evaluator records by typed ID and
epoch, verify uniqueness and aligned input/target, and report effective
serving/config differences. Use declared dataset IDs for prefixes. Report
fixes/breaks and parent agreement/disagreement strata; an oracle union is only
an analytical upper bound. Multiple epochs are not independent items for a
naive significance calculation. Prompt/termination diagnostics must inspect
actual token/finish evidence, not literal EOT suffixes. Test-derived records
remain outside scientist training/watch-set/failure-example inputs.

## 6. Predeclared outcomes and stop/selection rules

For every cell retain official score or explicit failure, time to first
validated incumbent, failed model-compute/repair/post-exit idle separately,
bookkeeping time, false blocks and exhaustive launch coverage, actual-count
and paired-evidence coverage, branch reuse, measured combinations and resolved
questions with resulting actions. Preserve unknowns and interval bounds.
Report every cell plus package spread; no scalar weighted composite, no
best-of-unequal-k comparison, no retrospective secondary-metric winner.

CPU release acceptance: zero known scope regressions, zero invented evidence,
all supported launch/save/close/export paths exercised, old and new consumer
tests passing, and independent forward behavior tests passed. GPU discovery
can expose new defects: withdraw only whole unstarted blocks if a confirmed
integrity defect makes them invalid; never cancel running work. A clean score
does not excuse false verification or test-data leakage.

Two discovery cells are mechanism/opportunity evidence, not proof of a stable
mean gain or equivalence. Continue only for a specific unresolved decision;
retain an efficiency/complementarity opportunity without calling it a quality
gain. There is **no promotion authorization** in this spec. A proposed
promotion needs a quality tolerance fixed before independent confirmation and
the reserved held-out task; do not choose the tolerance after seeing it.

## 7. Construction, test and scheduler dependencies

1. Implement in an isolated candidate worktree. Keep components separable for
   audit, but freeze/test the complete combined six-path snapshot.
2. Replay known real scope cases: two evaluator-only saves exemptions, original
   failing Trainer/merge and their repaired versions, all sampling-only fields,
   model-config migration, valid configs unchanged, mutation since early check,
   save/export faults and native CPU checkpoint/final-save routes.
3. Exercise E1×E2 eval-only lock, E2×E4 merge save, E3×E5 failed closure and
   Stop hook, E4×E8 exact selected serving settings, E6×E7 rendered versus raw
   tokens, K's relocated receipts and all collect/index consumers. For L/P,
   use duplicate/aliased data, out-of-order typed IDs, changed inputs/settings,
   partial logs, repeated epochs and retention promises as negative cases.
4. Independent reviewer receives realistic fixtures without the expected
   answer. Fix observed behavior, not merely textual snapshots. CPU tests do
   not establish GPU score gains or scientific retention/saturation claims.
5. Freeze accepted identities, new2-cell manifests and full site checks. Only
   then add receipt-backed held work under OWNERSHIP OK and exact frozen nodes.
6. Re-audit existing held blocks before any release. At08:06:53 UTC there were
   22 physical JobHeldUser cells and0 running, but the new policy does not
   certify all22 as still scientifically necessary. Do not release obsolete
   micro-screens/controls just to occupy GPUs.
7. Native two-node isolation and release authority remain required. Eight new
   discovery cells alone cannot supply16 running plus8 held. A6/8-cell wave
   also cannot be released from an8-cell-only useful buffer. Record this
   transition shortfall and prepare genuinely independent work; do not invent
   variants/repeats or count an unsubmitted manifest as held inventory.

Reusable construction lessons: [meta bundle validation](../../skills/exp_protocol_meta/bundle_validation.md).
