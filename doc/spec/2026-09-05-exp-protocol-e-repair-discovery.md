# E repair discovery after the first Opus4.8 comparison

User direction, 2026-09-05: skip GPQA, analyse the latest experiment results,
execute the next research round and use the 16 GPUs of this experiment subqueue.
The separately recorded pending-first direction releases already-valid work
while useful downstream work is prepared. No running or foreign work is cancelled.

## Evidence and treatment

The completed none/E block has six clean results, one flagged complete result
and one failed attempt. Clean means are63.1286% (none,n3) and54.9406% (E,n3).
This is descriptive, not a stable package-effect estimate. Preserve the flagged
E02 cost and the failed none03 separately.

E-repair is frozen at854464c677bdcc2bcf31fc504798f316e7dff8f7, protocol tree
de552c0555c42c50615c69dd28403da734ef08d4. Reference E isdcfa742, treeb3342236.
All six shipped paths and the new tree must pass the normal launcher check.

The package jointly addresses three evidenced interfaces:

- e02 future-data failure versus e04's successful sample/persist/new-training-card
  workflow: run CPU readiness before the engine factory, retain strict input hashes.
- e01/e03 unsupported native sampling calls and delayed failure: use the pinned
  native adapter and explicit bounded stages without global process cleanup.
- e02/e03 serving intent differs from supported fields: preserve selected JSON
  and actual request evidence; do_sample alone never proves greedy. No decoder
  is automatically changed and no recipe is mandated.

## Small, independently specified discovery

Two independent full-budget GSM8K scientist sessions, e04g01/e04g02. Preserve
the existing Opus4.8/high/1M route, Gemma-3-4B-PT revision,10h, one H100/16CPU/
128G and frozen images. Keep PTBdcf5da0 for this GSM8K comparison, independently
of ongoing HumanEval runtime construction. Actual scientist-selected recipes
are outcomes; this is not a fixed-recipe or precision ablation.

Two is the current limited discovery default for a new package; the original
four-replicate none/E block remains intact and is not renamed or rerun. At most
two extra cells require a written unresolved decision. No automatic null-control
top-up, no promotion, no held-out AIME2025 use, no GPQA dependency.

For every attempt preserve official score or explicit failure, judge/placement
status, actual feature use, false/unsupported blocks, prior-lock coverage,
producer-failure/repair/idle costs separately, and claimed versus supported decode.
Do not count wrapper-contained training time as paperwork. No score threshold
or post-hoc secondary metric establishes a winner at n2.

## Prelaunch evidence and limits

Changed-surface native CPU validation:91 passed, no skipped tests, using the
fixed Python3.10/vLLM0.11/Transformers4.57.3 and actual local tokenizer without
GPU devices. Ruff and syntax checks passed. The retained underscore skill name
has a known generic skill-validator naming rejection; it was not renamed.

A fresh-context independent forward exercise used only public skill/docs/CLI
and toy inputs. It chose separate sampling and training cards, did not fabricate
future data or model prerequisites, refused incomplete locks, and recorded the
do_sample-only decode as unknown. It performed no model call. This validates
behavior on that preparation scenario, not native engine startup, GPU teardown,
loadability or a score gain. Those are outcomes of receipt-backed discovery.

The optional broader23-module suite has no confirmed final result in the parent
context and is not counted as passed; the targeted91-test suite is complete.
Any subsequently reported failure is reviewed before dependent follow-ups.

## Execution and ownership

Submit this new manifest held, register the immutable receipt, and release only
after live ownership and each frozen ReqNodeList match. Scope is exclusively
gangda_exp-protocol-evolve on ondem0–1, at most16 allocated GPUs. Use the existing
operator queue/reconcile workflow and a per-receipt authorization record for
the shared reservation; never a global override or handcrafted sbatch.

The existing knowledge4/guard4 are already RUNNING. This independent discovery
does not wait for HumanEval admission. HumanEval remains a separate approved
cross-task study, with actual node/runtime/evidence checks required before launch.
