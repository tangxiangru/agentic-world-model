# HumanEval runtime/evidence checkpoint — 2026-09-04

Completed construction checkpoint, **not task/site admission or a model result**.
PTB source `09c90b63ad4f9daa2259ab0137ac323e0e345605` is committed and pushed to
`tangxiangru/PostTrainBench`, branch `codex/opus48-cross-task-runtime`.
Working clone: `/tmp/ptb-opus48-onboarding-oGLZFUf4/repo`, clean.
The operator's active PTB pin and all16 GSM8K receipts remain `dcf5da0`; no
existing receipt, method source or runtime was rewritten. No new job was submitted,
released or cancelled during this checkpoint. GPQA access still returns403.

## What is implemented

The shared HumanEval evaluator now uses a pinned, local-only parquet and the
original upstream mapper/instruction/generate/verify with explicit one epoch.
It does not call the upstream task factory's unpinned download.164 rows and IDs
were checked in a trusted CPU converter without showing or executing task code.
Reference/source identities are in the committed `data_provenance.json`.
Full selected content SHA256:
`7aeeca4daf4f06680efd86ea58b1d2233a100c2912e2618285fde80072f5494d`.
Metadata-only prompt/test AST inspection found static import roots
copy/math/random/string/typing, zero parse failures and zero explicit __import__
calls. Canonical solutions were not inspected for dependency design.

The common execution boundary replaces same-user `local` with individually
read-only public CPython3.10/NumPy files, isolated namespaces, bounded resources,
30s admitted wall timeout and owned-descendant cleanup. Exact requirements,
limits and remaining compatibility gaps live in PTB's
`src/eval/ptb_python_sandbox.md` and `tasks/humaneval/execution_contract.md`.
This is a changed shared execution contract, not a protocol treatment and not a
claim of old-environment/VM/kernel-exploit equivalence.

Each attempt retains request/selected-content bindings, model-file hashes,
runtime and materialization receipt, and full native Inspect JSON. Structured
execution evidence enters sample store, including exceptions and cancellation;
native Inspect's exception serializer otherwise loses custom report attributes.
The unchanged verify scorer maps ordinary exit0/nonzero and admitted timeout;
proven infrastructure/policy failures remain failed evaluations, not zero scores.

Full score publication validates typed IDs/epochs, actual counts, bound
input/target/test metadata, one execution/score per sample, executed-code hash,
clean teardown, runtime/helper/limits and recomputed accuracy/SEM. Model files
must be unchanged. An exact-byte raw snapshot is fsynced before one no-clobber
metrics commit. Original logger outputs and failed attempts are retained.
The host revalidates the independent frozen full selection, source, model and
runtime provenance, original/transport manifest hashes, receipt hash and limits.
Formal retry acceptance and the completed-run validator use that evidence;
the latter also accepts an independent expected task.

## Actual-image issue and correction

The original actual-image attempt failed **before code admission** with a
libbz2 file-bind EINVAL. The image root was fuse.fuse-overlayfs marked unbindable;
the same byte-identical file copied onto separately bound task storage mounted
successfully at both tested destination names. No isolation flag was removed.
The helper now materializes only its verified public closure, checks source and
copy bytes/modes/paths before each execution, and records transport separately
from original provenance. The image SHA is caller provenance; per-file bytes are
independently verified, not a claim of rehashing the whole image inside it.

Exact helper SHA256:
`94f41494e049d6a1e2e40c2057f25b205356eff72c343b13ef47e9b550035da7`.
Actual evaluation-image Python SHA256:
`7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86`;
the earlier host/opus extraction has the same version but different bytes.
At the final `/opt/ptb-python/bwrap` binding, original public Runtime identity is
`2d7606e28dfd14ed33e5940ce62ca0961319239a85ac85cde1147e57b65be2a7`.
The builder's earlier `/opt/ptb-bwrap/bwrap` path yields `01d60204…`; this is an
explicit source-path change, not evidence that transport overwrote provenance.

## Validation and informed review

- Exact backend:49 native host tests passed in185.33s. Original environment
  runs27 and explicitly skips22 native cases. Builder also reran13 actual-image
  records: public libraries, private visibility, concurrency, cleanup, original
  scorer C/I/I, and altered-copy/symlink/source-reference/non-overwrite rejection.
- Main's actual pinned vllm_debug image, final limits-bound wiring: four
  **invented** programs produced C, I, import-I, timeout-I through the original
  scorer and real backend. Every sample retained one structured report; native
  JSON → snapshot → publication → revalidation passed in47s. MockLLM only,
  no benchmark data, real model, GPU inference or training.
- A separate actual-image policy failure retained partial stdout and structured
  policy_denied/cleanup evidence, native status:error and no metrics. It is an
  expected rejection, never a scored scientist result.
- Host/pure evidence, completion, provenance and model regression:104 passed,
  22 explicit native skips. New full-chain tests use a tiny independently frozen
  **synthetic** profile/model, not fake HumanEval results. Shell syntax/diff checks
  pass; Ruff passes with the pre-existing non-executable evaluate.py shebang
  warning explicitly excluded, not silently fixed by changing task permissions.

Reviewer passes were **informed source reviews, not blind forward tests**.
They identified registration/output-loss defects, source-path and publication
durability gaps, and missing archive/limit content binding; main/builder repaired
and tested them. Main separately corrected the predecessor's stderr-keyword
classification, which changed native score semantics despite passing its old
test. Final review found the path/hash/limit corrections consistent with the
helper and no new direct blocker in that scope. Fresh-agent limits prevented
claiming a separate fresh-context guide forward.

## Durable raw evidence and next gates

[CPU evidence archive](humaneval-cpu-evidence-20260904.tar.gz),1,030,868 bytes,
SHA256 `2ba0bba9f70da104f90888663e039c08833f92a2b5f7b673b603f488238565ab`.
It contains original pre-admission failure, minimal mount comparison, exact
probe scripts, public manifests/receipt, native synthetic logs/metrics/snapshots,
error logs and metadata-only data check. It excludes public runtime copies,
models, credentials and benchmark rows. All original temporary folders remain.
Native repro commands are in the archived builder README; main uses the same
configured Apptainer with explicit read-only task/helper/bwrap binds, no `--nv`,
and task-owned TMPDIR. Do not call this compute-node/GPU-enabled acceptance.

Remaining before HumanEval receipts/admission:

1. Actual scientist-image and compute-node permissions/outer timeout/GPU-enabled
   containment tests; head-node CPU success cannot establish these.
2. Shared offline parquet and site bwrap provisioning, preserving dataset purpose.
3. Adopt the new common PTB pin and expected-task validation in receipt-backed
   AWM consumers, test discovery/judges/placement with synthetic bundles, and
   validate all three four-cell manifests/site constraints before held submission.
4. Separate release authorization/native ondem0–1 isolation and eight useful
   held cells remain mandatory. User has not yet approved a new reservation.

At14:16:46 ownership remained OK,0/16 allocated,38 held. Monitor PID3564003
wrote its14:16:40 tick,0/38 terminal, next15:16:40. No new Opus4.8 clean cell
or analysis window exists. Do not trigger another Claude trace synthesis from
these CPU tests. Generic skill quick-validation still rejects the historical
underscore name `exp_protocol_meta`; preserve the established identity. The
narrow reference update does not modify its frontmatter or user-owned drafts.

## Follow-up: actual scientist image and independent task consumers

At14:54, the exact same PTB09c90b6 backend ran inside actual `opus_5.sif` with
the scientist's contain/cleanenv/pid/no-init layout (CPU only, no --nv).
Five invented programs exercised C/I/import-I/timeout-I and private environment/
filesystem visibility. Native verify, sample store, exact-byte publication and
revalidation passed in52s. Original public runtime is
`8f0ce4b3c4702f2d8b497cd35dd6f42322c71af04a6fa906ea423ad790cc515c`,
distinct from the evaluation image; helper/declared limits are unchanged.

The AWM consumers now pass task identity from the manifest or exact receipt
job/cell. A naked result directory is diagnostic, not authority. Old GSM8K/AIME
retain the legacy validator CLI plus AWM's task-field check; HumanEval requires
the new --expected-task interface, with no retry under a weaker interface.
Manual harvest resolves a unique matching batch/cell/job receipt; missing or
conflicting identity preserves evidence but cannot mark complete. The old PTB
pin remains active, so this is not an allowlist bypass.136 focused tests pass.
Actual historical guard remains8 clean; strict-control remains7 clean plus
one placement quarantine after the change. No historical score was changed.

Compute-node access: direct SSH initially had no trusted host key. Read-only
GCP API lookup identified ondem0 in sercan-v1/us-west1-a and provided its
hostkeys guest attributes. A task-local known_hosts pin allowed strict server
identity verification without editing global trust. The default identity and
the already-existing google_compute_engine identity were both refused for
robtang_google_com (publickey). No identity was generated/uploaded, no login
policy or node configuration was changed, and no weaker host check was used.
Node execution/--nv/outer-timeout acceptance therefore remains unproven and
requires authorized access; no GPU/native-reservation authorization is inferred.

[Scientist-image raw archive](opus-image-cpu-evidence-20260904.tar.gz) contains
the synthetic logs/snapshot/runtime, probe and public host-key provenance,
not private credentials, actual models or benchmark items. The separate
reservation and GPQA-access requests remain unanswered.
Archive321,586 bytes, SHA256
`916edf68c190b9b11711b87b4cb10daf017d6d78038994f321f74c2346ed78a2`.

The [shared-data preparation](shared-data-preparation/README.md) adds verified
offline shared-cache reads in both actual images, native concurrency-default
checks,12 explicitly unsubmitted HumanEval drafts and the current node-login
diagnosis. It does not waive compute-node admission or native release gates.
