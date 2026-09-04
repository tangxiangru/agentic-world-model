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
