# Bundle construction checkpoint — 2026-09-04

This records **partial construction, not a launched round or completed E**.
The [bundle spec](../spec/2026-09-04-exp-protocol-bundle-discovery.md) was committed
first as4b14768. Construction is isolated in
`/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`, branch
`codex/exp-protocol-bundles-20260904`. The operator's runtime tree is unchanged.

## Variants

| label | source | change |
|---|---|---|
| reference |guard drift2f64581 / protocol tree189319d6|historical reference|
| E in construction |commits carrying this record; not an approved frozen variant|H/J/K integration, honest comparator evidence, tested native-save adapter and foreground execution records|
| E+L / E+P / E+L+P |not constructed|distinct additions from the predeclared spec|

This checkpoint incorporates H `b52e5f2`, J `549e25a`, K `58a6992`, B `9f294c3`
and the useful process guidance from E2 `c6f11d8`. It does not import D1's
unconditional parent-config blocker or E1's unchanged-tail death inference.
K's full receipt/lock/index/collect/Stop-hook consumers are carried together.
The previous E2 source's claimed strict idle lower bounds are corrected, not
reused as proof. B's historical stop IDs are examples, not universal constants.

Repeat spend: zero new jobs/attempts; no standalone H/J/K/B funding. Final
packages retain the2-cell discovery and decision-specific≤2 extension budget.

## Cells

Planned outer contract remains GSM8K, Gemma-3-4B-PT, scientist Opus5[1m]/high,
1M context,10h and the same frozen PTB/images. No new cells, manifests or
receipts exist. Held-out task: none. Synthetic CPU fixtures are not scientist
cells, benchmark scores or new-clean observations.

## Results

No GPU outcome vector yet. Current official record remains g01r03's82.79%,
which is unrelated to the effectiveness of this unrun package.

CPU evidence through09:18 UTC:

-36 original K lifecycle tests passed after the first mechanical integration.
-183 protocol/sandbox tests passed after H/J/K and revised guidance integration.
-196 passed after13 legacy comparator evidence cases.
-202 passed after6 H×K completed/failed non-training lifecycle cases, including
  no invented data, unchanged locked plan, blocked unverified conclusions,
  failed closure without fabricated comparator output and all consumers.
-100 PTB manifest/operator tests passed after preparing the pinned submodule
  and shared data path in this new worktree.
-51 save tests passed in the extracted pinned-image CPU runtime:32 stand-in/
  control cases and19 actual native cases. The main planner independently
  replayed the exact command from `skills/exp_protocol/save-safety.md`, obtaining
  `51 passed in10.60s`. Real tiny GPT2/Gemma3 saves, single-device CPU reload,
  Trainer checkpoint/final/state-dict paths ran without forwards or training.
  The original Python environment explicitly skips native cases because it has
  no Torch/Transformers; its passing stand-in tests are not the native proof.
  The combined original-environment protocol/sandbox suite also passed after
  routing the scientist skill to the new helper, with these expected skips.
-31 E5 execution tests passed in the original environment and independently in
  the pinned image's Python3.10 runtime (`31 passed in2.98s`), without models,
  network or GPU devices. They exercise real child exits/signals/observer death,
  fresh namespaces, stale inputs, same-card concurrency, incomplete receipts,
  allowed result-only card updates, and E3 failed closure.
-Final original-environment combined protocol/sandbox/PTB suite:
  `365 passed,15 skipped in18.84s`. The15 skip reports are the known missing-Torch
  native save cases, not unknown E5 failures. Save's independent51-test pinned
  evidence above remains separate.
-Independent [E5 forward review](trace-reviews/bundle-e5-forward/report.md): six
  real CPU attempts, seven honestly closed cards, no locked-open cards and no
  remaining owned child processes. Seventy original text artifacts were archived
  byte-identically, including unsuccessful/interrupted/unknown attempts.

The generic skill-creator validator exits1 on the repository's preexisting
`exp_protocol` underscore name. Preserve that installed identity; do not rename
it to satisfy an unrelated naming convention. Behavioral/repository tests and
reference routing are the validation here, not a claimed generic-validator PASS.

The first broad PTB run had96 passes/4 failures: the new worktree lacked its
pinned PTB checkout and shared HF data path. Those were real setup deficiencies,
not protocol failures. Locally cloned PTB atdcf5da0, correct fork/upstream
remotes, private ignored site configuration and a shared data symlink resolved
them. The test-created180KB AWM snapshot cache was moved intact to the parent
`test-generated-data-cache`; no experiment artifact was deleted.

Commands use the original operator `.venv/bin/python` with explicit
`PYTHONPATH=/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo` and this
worktree as cwd. `awm.__file__` and `paths.REPO_ROOT` were checked and resolve
here. Tests: `tests/test_exp_protocol*.py tests/test_sandbox.py` and separately
`tests/test_ptb_experiments.py tests/test_ptb_ops.py`.

## Trace review

Reuse the closed Window04 and focused opportunity review; do not re-count
them as new cells or launch another full Claude synthesis for this checkpoint.
The raw card/trace evidence and three-card reads are preserved in the linked
Window04 decision. Save-specific review added the native model-config migration
edge before serialization. E4's implementation and real pinned CPU tests now
cover that edge and actual Gemma3 config/model behavior; this still does not
establish scientist adoption, universal save coverage or an experiment effect.

## Directions

The shared operator directions ledger has concurrent user edits and is not
staged by this checkpoint. The spec's component ledger records current scope.
J/B/H/K are integrated components, not four standalone GPU screens. Old held
blocks remain physical holds pending scientifically justified replacement;
native isolation/release authority remains unmet.

## Decision

Keep the current baseline; continue constructing the **complete** E package.
No submission, release, promotion, quality gain or restored useful-held floor
is claimed. Eight new discovery cells alone would not satisfy16 running+8 held.

E5 is accepted as a tested component, not a full E package. E7's
[rendered-input interface](2026-09-04-rendered-input-design.md) is accepted for
construction after original g03/g05/g08 source/trace checks; no token materializer
or consumer is implemented yet. These CPU reviews add zero validator-clean cells.

## Change

Legacy comparator files with only requested limits, scalar accuracy/stderr or
unsupported/non-JSON content are now unverified warnings rather than PASS.
Conflicting actual counts remain failures. Missing optional expected metric
does not become a new required schema field; the missing verification is
visible. Structured Inspect sample arrays are counted instead of compared
directly with an integer. Full identity and target-file verification remain
outside the inherited K proof; minimal summaries are not independent audits.

The Stop hook combines actual-producer/exit guidance with K's stricter receipt
closure; manual conclusion filling cannot certify a deferred comparison.
Optional WMA calls require installed command availability and never block work.
Unsupported universal90%-execution attribution is removed. Export guidance
separates CPU metadata checks, locked serving probes and selected on-disk decode.

E4 adds `GenerationSaveContract`, `SaveSafeTrainer`, tests and the linked
`save-safety.md`. It projects actual in-memory migration, narrowly normalizes
invalid greedy serializer copies, preserves original objects and exact selected
serving JSON, checks source/API identity, and records failures without replacing
the original exception. The main review tested two potential gaps: pinned4.57.3
has no older contrastive top_k exception (regression pinned), and `1e999` JSON
overflow is now rejected. The actual historical whole-model `device_map="cpu"`
is supported and tested; sharded/offloaded/custom/distributed paths remain
unsupported. The skill routes save operations to this contract, not eval-only
cards to a global parent-config blocker.

E5 adds `awm exp_protocol run`, private no-clobber attempt records, exact argv/cwd
and declared environment application, live lock/script/data checks, owned-group
signal handling and optional predeclared fresh-directory evidence. It never
auto-closes, retries, selects a model, or equates exit0 with scientific completion.
An unresolved prior observer-death attempt cannot be retried solely because the
OS lock disappeared; current PID birth identity is reported, not guessed.

Main review found and reproduced a false block on legitimate result-section
writes: the correction checks the immutable plan/card identity/lock, while
allowing sections5–6 to be recorded after observing results. Absolute data paths
are required by this optional wrapper because v2 locks did not pin the caller
cwd for relative paths. Interrupted cleanup now timestamps an observed child
exit; an expensive partial-output inventory is not forced and stays unverified.
The independent reviewer tested the pre-correction execution source and did not
rerun after it was unfrozen; the parent's final regression tests prove these
specific subsequent corrections. No blanket claim of a blind final-code audit.

## Evidence

See spec components E1–E8 and Window04 D scope, card semantics, exit/timing and
paired-count audits. The new tests intentionally cover unknown evidence,
conflicts, absent helpers and combined consumers, not only matching headings.

## Next round

Integrate E4/E5 evidence with the remaining selected-artifact/export consumers;
implement E6 real sampling helpers and E7's prepared-token/checked-consumer
interface. Validate the whole package with independent forward cases.
Only afterward freeze complete variants and2-cell manifests, prepare L/P,
replace scientifically obsolete holds through exact receipts, and revisit
release with real native isolation/authority. A passing partial suite cannot
substitute for those remaining components.
