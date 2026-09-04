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
