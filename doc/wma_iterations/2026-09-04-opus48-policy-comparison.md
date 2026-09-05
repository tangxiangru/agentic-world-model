# Opus4.8 old/new single-card policy comparison

Status: preparing the preregistered S0 reference archive and technical validation;
four scientific cells will remain staged until exact production acceptance.

The user requested newer-method comparisons on GSM8K and other benchmarks.
The existing five-arm study does not separately measure old versus new WMA
policy, so one additional four-cell setting holds the new public/private
runtime, Opus4.8 models, single-card mode, budget and benchmark fixed while
restoring only legacy v0.2 policy text. It compares with existing S/w57r01–04.
This decision uses no new-study result and does not wait on unrelated H tails.

Contract: `doc/spec/2026-09-04-opus48-wma-policy-comparison.md`. Record source
freeze, skill hashes, validation receipt and outcome, scientific manifests/jobs,
costs and exclusions below as they become available. This is no promotion and
does not replace either failed raw baseline attempt.

## Legacy reference archive

This archive commit restores exactly the three v0.2 policy files from ae46724;
its hash is **176f0a464986**. Every non-skill private shipped path is identical
to the new runtime31b854bb. All six existing skill-contract tests pass. No new
rule is learned or promoted here. The current operator policy17be8a23046a will
be restored before submission; manifests will explicitly pin this archive.
File hashes and checks are in `evidence/2026-09-04-policy-comparison/`. Actual
production acceptance of this public/private/model combination remains a real
launch dependency. The scientific S0 cells are not yet submitted.

## Frozen manifests and real acceptance dependency

Legacy private archive: `f8e4c2284f4da82be961aca3a15dfa61ca6fd765` (skill176f0a464986). Common public
source remains31b854bb. The operator policy has returned to17be8a23046a.
The scientific manifest is `wma-crossbench-opus48-r06-gsm8k-legacy-v02-x4.yaml`, w66r01–04,
explicit single-card mode, staged with no runtime acceptance path.
Technical manifest `wma-crossbench-opus48-legacy-v02-runtime-validation-x4.yaml` submits only v91r01.
Its unique context record is `data/ptb/context-validation/legacy-v02-f8e4c2284f4d-opus48-high-200k.json`; existing queued receipts
retain their original proof file. No common runtime or scorer changed.
The exact technical receipt/job and acceptance artifact will be recorded next.

Source/site checks return zero issues for S0 and for validation v91r01. The
full scientific gate correctly remains blocked by the missing exact runtime
acceptance and its separate200k context proof. Both legacy and restored current
policy pass all six existing skill-contract tests. Non-skill private source is
byte-identical to31b854bb. No missing gate is bypassed; the next action is the
registered validation-only job, with a30-minute cap and independent proof path.

## Validation submitted asynchronously

Technical-only **v91r01 / job92312** is PENDING on the exact nodes2–3 route.
Receipt: `results/ptb/wma-crossbench-opus48-legacy-v02-runtime-validation-x4/context-smoke-1-2026-09-04T180034.371470+0000.json`.
It was registered and route-checked while held, then released. Source top is
9de9865a; public31b854bb; privatef8e4c228; original context proof is untouched.
No scientific S0 job is submitted; all four remain staged.

The PTB worktree advanced concurrently from e62036f to **9695e32f3a160b8c7e927cbb4de66727de7c72ad**
between preview and submission. The actual receipt freezes the latter, while
the top gitlink remains e62036f. The only changed file is standalone
`src/judges/smoke_claude_vertex.sh`, not called by this experiment; all launch,
agent, acceptance, task and evaluation code is unchanged. The exact diff and
limits are recorded in `ptb-source-equivalence.json`; keep this external work
and do not silently stage its submodule pointer. Recheck equivalence before the
scientific launch; future relevant changes are not automatically approved.

Continuation: inspect job92312 and the acceptance JSON under
`data/ptb/context-validation/wma-runtime/f8e4c2284f4da82be961aca3a15dfa61ca6fd765/92312/`.
Require passed context/model/isolation and matching public/private SHA. Then
freeze its exact path in the S0 manifest, full-check, inspect reconcile preview,
commit/push and submit w66r01–04 without waiting on unrelated science. A failed
validation remains a blocker; it is not a scientific failure or a promotion.
