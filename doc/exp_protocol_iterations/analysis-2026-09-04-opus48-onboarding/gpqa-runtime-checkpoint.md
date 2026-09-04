# GPQA software checkpoint — 2026-09-04

Independent CPU construction is complete at PTB
`60df491f2bac85ceb801d8b04b706023ce76e02c`, pushed to
`tangxiangru/PostTrainBench`, branch `codex/opus48-cross-task-runtime`.
The construction clone `/tmp/ptb-opus48-onboarding-oGLZFUf4/repo` is clean.
This is **not real-data acceptance, task admission or an experiment result**.
The active operator submodule and all16 GSM8K receipts remain PTBdcf5da0.

## Implemented and reviewed

The source separates PTB task `gpqamain` from native Inspect `gpqa_main`, loads
a pinned local CSV only after a frozen data-profile check, preserves native COT
multiple-choice/choice scoring and unseeded population shuffling, and records
source options by original position. Shadow markers exercise the native
permutation without confusing duplicate answer strings; markers never reach
the model. Actual source bytes, population, typed IDs, ordered rows and
contamination reference remain unfrozen: the official source is inaccessible.
No fabricated real profile or reference is committed.

Each evaluation retains an independent pre-generation request and native JSON,
with sample/content/presentation/configuration and model/source/template bindings.
Verified raw bytes are durably snapshotted before no-clobber metrics. GPQA uses
its own choice evidence, not HumanEval's executor. Formal retry phases remain
16k×4/12k×3/8k×2, full population, one epoch, concurrency6 and memory fraction0.8.
AWM and PTB require independent expected-task checks for both new tasks, with
no fallback from a rejected strict check to the legacy interface.

Informed reviewer `humaneval_forward` found a real gap in the first version:
changing generated text while retaining C/answer, or removing model output,
still passed. The fix binds successful model-event output → retained completion/
assistant → native single-answer parse → score/explanation. The runtime parser
and scorer are unchanged; the host stdlib projection is differentially tested,
including case sensitivity and first-match behavior. The reviewer revalidated
both original synthetic logs and confirmed both attacks now fail. This was
informed source/evidence review, not a blind forward test.

HumanEval received a narrow compatibility fix: developer evaluation can hash
existing HF **file** symlinks, and a provided base ID resolves only to its declared
revision snapshot. Formal final-model validation stays strict; directory links
and unpinned downloads are not enabled. Its sandbox helper remains
`94f41494e049d6a1e2e40c2057f25b205356eff72c343b13ef47e9b550035da7`.
The12 HumanEval drafts now require the new common PTB checkpoint but remain
unsubmitted, unadmitted and outside the held count.

## Actual checks and evidence

- Pure PTB regression:168 passed,24 explicit native skips. Includes synthetic
  validator and AWM discovery/harvest clean, anomaly and placement-only cases.
  Operator focused regression:140 passed.
- Both pinned images:300 native shuffle seeds each, all24 permutations,672
  native parser combinations each. Seven invented mock samples cover C/I,
  identical-text wrong option, missing answer, lowercase, empty output and
  reasoning/text parts. Publication/revalidation and both output-tampering/
  deletion negatives pass. No real model, GPU or benchmark data.
- Both images' actual evaluator CLI rejects the missing real profile before
  model resolution, with no metrics or model log.
- Ruff and shell syntax/diff checks pass (existing evaluate.py shebang EXE001
  explicitly excluded). A later optional native pytest invocation found no
  pytest installed in the image: those two pytest cases were not run there.
  The standalone probe above actually executed their mechanism checks; the
  image was not modified to hide the missing runner.
- Generic skill quick-validation still rejects the existing underscore name
  `exp_protocol_meta`; its established identity/frontmatter and unrelated user
  drafts were preserved. The narrow lessons are in the existing cross-benchmark
  reference, linked through the operator state and bundle-validation guidance.
- One successful outer Apptainer run emitted a fuse-overlay cleanup INFO about
  connection abort after probe success, exit0. This is an outer-container
  observation, not compute-node/timeout acceptance.

[Raw CPU archive](gpqa-cpu-evidence-20260904.tar.gz):94,434 bytes,
SHA256`f7577a4247000c017fde8b5c0f29af81edf838cb1c48543f89945bd1ca9b51d9`.
Contains probes, independent requests, native synthetic JSON, metrics/snapshots,
summaries and missing-profile stderr. The earlier five-case probe remains
separate from the corrected seven-case run. No credentials, model weights or
real GPQA items are included. Original temporary artifacts remain.

## Actual remaining gates

17:31 UTC checks: OWNERSHIP OK;0/16 allocated; all92125–92140 remain
PENDING(JobHeldUser).16 useful new holds plus5 untouched legacy holds. The
reservation still covers11 nodes, not native isolated ondem0–1. Existing GPQA
credentials still receive GatedRepoError/403 for official revision metadata.
No new reservation/IAM/login-policy/data-access authorization has arrived.
The last node SSH check at16:21 used the trusted server key and matching
existing OS Login identity and was refused; it was not bypassed.

Monitor3684437 remains live, hourly;17:16:40 state has0/21 terminal and next
scheduled check18:16:40. No new Opus4.8 clean cells or Claude analysis window.
Highest historical clean GSM8K remains1092/1319=82.79%; this checkpoint does
not improve or update that score.

These external gates have persisted for at least three goal turns. GPQA's
independent synthetic integration/review is now finished. Further meaningful
admission/execution requires authorized official data and compute-node access
plus native two-node isolation/release authority. Keep the detector and useful
holds intact, mark the planning goal blocked once this checkpoint is durable,
and request direction; do not fabricate profiles, task admission or GPU occupancy.

On authorization: compare actual CSV/null/types with the original loader;
freeze lawful source/reference/profile; perform compute-node/GPU-enabled and
outer-timeout acceptance; adopt the common validated PTB source; fully validate
and admit matched new-task arms before receipts. Release only with ownership,
frozen nodes, native isolation and at least eight useful cells remaining held.
Do not reuse the old90791–90798 exception or rewrite existing GSM8K receipts.
