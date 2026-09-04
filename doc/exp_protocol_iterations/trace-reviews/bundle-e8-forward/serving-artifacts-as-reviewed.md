# Freeze and publish the selected serving artifact

Use these helpers to preserve an already chosen artifact and decoder. They do
not choose a checkpoint, change generation settings, launch inference, or prove
quality. Keep model execution/evaluation under its prior matching card and lock.

## Freeze before relying on a measurement

~~~python
from awm.exp_protocol.serving_artifacts import (
    snapshot_serving_artifact, verify_serving_artifact, publish_serving_artifact,
)

selected = snapshot_serving_artifact(source)
# Retain this manifest outside the model directory, e.g. memory/selections/.
# Obtain the hash from the scientist's already-frozen selected generation bytes.
# With E4's explicit serving export, use its selected_serving_hash, NOT the
# serializer-only serialized_file_hash.
generation_sha = frozen_selected_generation_sha256
verify_serving_artifact(source, selected, generation_sha)
~~~

Freeze that expected identity before the evaluation whose result you rely on,
and verify against the SAME identity afterward and before publication. Preserve
the evaluator command/contract and its own before/after evidence separately.
A snapshot made after evaluation proves current bytes only; it cannot establish
what the earlier evaluator read. Do not replace a missing expected hash with a
hash of whichever file happens to exist now.

The manifest records a versioned profile plus every selected serving file's
relative path, byte size and SHA256. Its content identity is independent of the
directory label, so staged and published copies can be compared exactly. Keep
the caller's task/reference/measurement provenance alongside it.

## New destination by default

~~~python
record = publish_serving_artifact(
    source, destination, selected, generation_sha,
    session_dir=task_directory, task_id=task_identity,
    reference_id="exp-07-selected-artifact",
)
~~~

The destination must be a narrow directory inside the explicit existing task
scope, not the task/workspace/home/root itself or a broad data/memory target.
Source and destination must not overlap. The destination parent must already
exist; unexpected ancestor aliases are rejected. An existing destination is
never emptied, merged into, or silently replaced.

E8 creates its own stage, copies only manifest-listed serving files, verifies
the complete stage and rechecks the source, then uses Linux
renameat2(RENAME_NOREPLACE). A destination appearing during publication causes
failure rather than being clobbered—even if it is empty. Unsupported filesystems
get an explicit error; there is no unsafe rename fallback.

Do not have E5 reserve this destination with fresh-directory: E8 needs an
absent destination for its rename. E5 can record the earlier producer in a
separate namespace or use its documented unverified-output mode for publication.
E5 exit/freshness evidence and E4 save evidence do not become quality certificates.

## Explicit recoverable replacement

Replacement is allowed only after the caller has established that the owned
target is quiescent: no evaluator/trainer or other consumer is using it.
Do not infer this from GPU memory, a shared Unix user, a missing PID, or a single
direct-child exit when descendant/other-consumer ownership remains unknown.

~~~python
record = publish_serving_artifact(
    source, destination, selected, generation_sha,
    session_dir=task_directory, task_id=task_identity,
    reference_id="exp-07-selected-artifact",
    replace=True, expected_old_identity=frozen_incumbent_identity,
    target_quiescent=True,
    quiescence_evidence="All known owned consumers relinquished the target; see retained ownership/wait evidence.",
)
~~~

The flag/text RECORD the caller-established condition; the helper does not
independently prove it or stop processes. The old target must match its separately
frozen identity. It is rechecked immediately before replacement and retained in
a unique sibling backup. The operation never recursively deletes an incumbent,
backup or failed stage. Backups consume storage until an independently authorized
retention decision; this helper does not prune them.

## Journals and recovery

Each operation records its task/reference, frozen identities, intended stage,
backup and failed-publication paths, phases and verification results under
memory/serving-publications/. An OS lock excludes competing guarded publishers
of the same destination; the lock file is not proof that a publisher is alive.

The backup/publish pair is recoverable, NOT an atomic directory exchange. There
can be a window with the old target at its backup path and no destination.
Catchable failures attempt rollback only through the identified directory
objects and no-replace renames. An unexpectedly occupied destination is not
reclaimed. Failed copies/stages remain available; a failed new publication is
retained under its recorded failed path when safely retractable.

For ServingPublicationError, inspect exception.report and its record_path.
An interrupted Python call preserves the original interruption with
exception.publication_record when cleanup ran. Rollback failure is separately
recorded as requiring manual recovery; never call it a restored incumbent.
SIGKILL, unhandled termination or host failure cannot guarantee cleanup/final
journaling. Inspect the last phase and the recorded paths/identities before any
retry. Do not blindly delete stages, overwrite a new occupant, or assume path
existence means publication completed.

## Supported verification layers

- Profiles: current Gemma3 multimodal (Gemma3ForConditionalGeneration) and
  native GPT2/Llama/Gemma3-text causal-LM layouts. Gemma3 requires both
  preprocessor_config.json and processor_config.json.
- Required common assets: config, generation config, fast tokenizer JSON/config,
  and either one standard weight file or a complete native shard index. Known
  tokenizer vocabulary/merges/special-token and processor/template files are
  included when present; optimizer, training-state and recognized logs are not.
- Safetensors checks cover header dtype/shape/offset ranges, payload coverage,
  shard completeness, tensor-to-shard maps and declared total sizes. They do
  NOT establish model-parameter alignment, tensor-value correctness or quality.
- Pickle .bin weights require allow_opaque_weights=True explicitly at
  snapshot/verify/publication. They remain opaque byte identities; no pickle is
  executed and no weight correctness is asserted.
- Verification/publication defaults to real local metadata loaders from pinned
  Transformers 4.57.3, with no network or remote code. It loads configuration,
  tokenizer, generation and (for Gemma3) processor metadata, never a model.
  Explicit load_metadata=False is reported as structural-only verification,
  not silently promoted to a successful native metadata load.
- Mixed/unknown layouts, missing or empty shards, path traversal, symlinks,
  hardlinks, special files, remote/custom/quantized metadata and unknown required
  assets are unsupported or invalid. Flat layouts only; named-template
  subdirectories and other new layouts need a tested adapter.

No mode rewrites temperature, EOS, cache flags or other selected settings.
A normalized-but-unselected E4 checkpoint is refused when its generation hash
differs. CPU metadata success is not vLLM loadability, successful evaluation,
contamination checking or PTB scientific completion. Publication is not a
security boundary against privileged/uncooperative namespace manipulation;
caller ownership, quiescence and exclusive publication coordination still matter.

## CPU construction checks

Run pytest tests/test_exp_protocol_serving_artifacts.py. Dependency-free tests
exercise byte/layout/copy/recovery behavior. Pinned tests use real local metadata
loaders, tiny native E4 saves, sampled/greedy and indexed shards, Gemma3 processor
metadata and a real E5 CPU execution record. No model forwards, training,
evaluation, real benchmark weights, downloads or scheduler actions are involved.
The pinned bwrap setup is in save-safety.md; select the serving-artifact test
file (and save/execution tests for the combined pass).

A local 16 MiB synthetic payload measured about 0.030 s snapshot, 0.026 s
structural verification and 0.16 s publication. These include content hashing
and copying but exclude native metadata loading; they are not an extrapolation
to full model sizes, a serving benchmark, or a scientific gain.
