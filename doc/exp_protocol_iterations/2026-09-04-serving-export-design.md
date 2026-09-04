# Selected serving artifact: E8 construction contract

2026-09-04. Construction design, not a launch or a new experiment result.
Source: g01r05's retained `task/scripts/finalize.py` removes an existing final
directory before copying, then conditionally rewrites generation settings to a
fixed greedy spelling. Its CPU model load is not proof of unchanged selection
identity or actual vLLM behavior. E4 already separates safe serialization from
the scientist's selected serving JSON; E5 distinguishes fresh output evidence
from semantic validation. The remaining consumer must preserve those distinctions.

## Required interfaces

1. `snapshot_serving_artifact(path)` produces a versioned content manifest of
   the selected supported HF serving files: model config, generation config,
   tokenizer/processor assets and the actual weight file/shards/index. It records
   paths, sizes and SHA256, not just mtime or a reused directory label.
2. `verify_serving_artifact(path, expected_identity, expected_generation_sha256)`
   checks the actual files against that already-frozen identity and selected
   serving settings. It checks JSON/index structure and required local metadata,
   with an explicit supported loader/profile and no model forwards or downloads.
   A serializable checkpoint is not automatically a selected serving export.
3. `publish_serving_artifact(source, destination, expected_identity, ... )`
   stages a complete verified copy before changing the destination. Default is
   a new destination only. Explicit replacement, when needed, preserves the old
   target under a unique backup and records the operation; no recursive deletion
   of the previous incumbent, and no silent merge into a populated directory.

These tools do not select checkpoints/decoders. They must never rewrite
temperature, EOS IDs, cache settings or other serving parameters to a preferred
recipe. Caller-provided expected hashes must be frozen before relying on a
measurement; snapshotting after evaluation cannot retrospectively prove which
bytes were measured. Preserve the evaluator invocation/contract and its separate
artifact before/after evidence where available; missing identity remains unknown.

## Identity, scope and mutation rules

- Bind all files actually used by the supported layout, including every shard
  named in its weight index. Reject missing/empty shards, malformed maps and
  ambiguous mixed weight layouts. Do not deserialize arbitrary pickle merely
  to call a `.bin` file valid; distinguish opaque-byte identity from structural
  safetensors checks and from a real model-load result.
- A local tokenizer/config/processor load is CPU metadata evidence only. It does
  not prove inference-engine compatibility or weight correctness. Any actual
  load/forward/evaluation probe still runs under its prior matching card/lock;
  PTB scientific completion still requires the unchanged official validator.
- Tokenizer and processor requirements are explicit for the selected model
  profile. Unknown/custom/remote-code layouts are unsupported, not silent PASS.
  No network fallback or arbitrary remote-code execution during metadata checks.
- Model outputs from E4's supported native serializer and ordinary regular
  serving files are the initial publication target. Symlinks/path traversal,
  special files and mutable files encountered during snapshot/copy are rejected
  or require an explicitly scoped adapter; never follow an unexpected link into
  unrelated data. Do not treat a Hub cache alias as a newly produced artifact.
- Do not copy optimizer/logging state merely because it shares a training-root
  directory. The supported serving manifest determines the copied files; unknown
  required inference assets must not be silently omitted.
- Verify the staged copy and recheck the source identity before publication.
  A failed copy or source mutation leaves the existing target untouched and the
  incomplete stage available for diagnosis. Record both paths and error state.
- Replacement requires a quiescent owned target: no running evaluator/trainer
  may be using it. The helper does not infer this from GPU memory or a Unix user,
  stop processes, or claim a caller declaration proves quiescence. A two-rename
  backup/publish sequence is recoverable but is not an atomic directory exchange;
  preserve and document the interruption/recovery boundary honestly.
- E4 save receipts certify their observed native save scope, not an entire
  Trainer checkpoint or measured model identity. E5 fresh-directory snapshots
  certify namespace/file observations, not quality. E8 consumes their evidence
  without upgrading either into an unearned model/benchmark certificate.

## Required CPU tests before the component is complete

Use tiny native/synthetic fixtures, never a new benchmark/GPU run. Cover exact
selected generation bytes through safe-save and publication; sampled and greedy
settings both preserved; normalized-but-unselected checkpoints refused; missing
tokenizer/processor/config; index/shard completeness; mixed layouts; unexpected
links/path traversal; changed source during copy; staged-copy faults; destination
already populated; explicit recoverable replacement and interruption/rollback
failure; previous incumbent retained and no unowned target touched.

Test the real local metadata loaders in the pinned environment without weights
where possible, and separately label any tiny native weight-serialization test.
Use E4/E5 records from real CPU fixtures for combined tests. Do not count a
passed metadata check, a fake model file or a unit test as scientific completion.
Keep publication ownership and planned reference/task identities explicit.

This contract must be implemented and jointly tested before declaring E complete.
It does not authorize a Slurm release, a new repeat, or a protocol promotion.
