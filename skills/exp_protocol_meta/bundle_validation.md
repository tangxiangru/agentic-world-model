# Validate a combined protocol, not a bag of passing checks

Use with `search_policy.md` when constructing a coherent package from trace
findings. The outer bundle policy does not change the scientist's experiment
unit or waive card/check/lock. Source-specific mechanisms and candidate details
belong in the [bundle specification](../../doc/spec/2026-09-04-exp-protocol-bundle-discovery.md).

- Keep scientific policy transitions separate from operational permission.
  An explicit newer human search decision supersedes historical defaults;
  asking for the same strategy approval again is not an operational gate.
  It does not grant native infrastructure changes or Slurm release exceptions.
- Old held jobs retain physical scheduler state, not perpetual scientific
  priority. Re-audit whole blocks for current decision value before release.
  Report physical held and currently justified/releasable inventory separately.
  An8-cell discovery design is not16 running plus8 held; do not manufacture
  repeats to hide an inventory shortfall.
- Preserve whole-package scope. A tested first component is construction
  progress, not permission to rename it as the full intervention. Reuse earlier
  tested components but re-test their shared consumers and interactions.
- Validate the actual operation at its boundary. Static parent artifacts and
  family labels are useful hints, not proof of effective in-memory behavior.
  Native code can migrate configuration immediately before save. Pair early
  cheap checks with the actual supported save path; include eval-only, merge,
  in-code repair, intermediate checkpoint and final-export cases.
- Restoring in-memory settings does not restore serialized serving settings.
  Preserve selected on-disk bytes/hashes separately from safe checkpoint
  serialization. A wrapper's success or import does not prove bypass-free
  coverage; audit actual callers and label unsupported paths honestly.
- Export verification needs the already-selected identity, not a fresh snapshot
  taken after an undocumented decoder change. Native metadata loading, weight
  file structure, exact serving bytes and an actual evaluator result are
  separate evidence. Test both sampled and greedy selected settings, all indexed
  shards and the actual model profile's tokenizer/processor assets.
- Stage a complete selected export before replacing an incumbent. Recoverable
  backup/publication renames are not an atomic exchange; record intended paths
  before mutation and retain failed stages and the old artifact. Quiescence is
  an externally established precondition, not something a flag proves. Test
  interruption and a competing unowned destination without reclaiming it.
- Scope metadata checks to metadata: a vocabulary token named `tokenizer_file`
  or `custom_generate` is data, not a configuration directive. Native tokenizer
  serialization can retain both vocab/merges and tokenizer JSON; exercise that
  actual layout instead of generalizing from a minimal synthetic tokenizer.
- Missing evidence must stay unverified across old and new consumers. A
  legacy comparator's existence, requested limit or scalar/SE pair is not a
  verified completed count. Mark unsupported evidence explicitly rather than
  turning it into PASS or inventing a new required schema field. Test absent
  helpers and combined non-training/deferred-failure closure, not just isolated
  positive JSON examples.
- Update resume/wake-up guidance when analysis policy changes. The terminal
  detector does not establish scientific completion; a completed predeclared
  small discovery block can trigger review without filler to reach eight.
  Preserve the live monitor's cumulative IDs and cadence while changing that
  guidance; a text change is not grounds to restart a healthy detector.
- Use independent forward tests on realistic inputs, not only regressions
  that encode the intended answer. Keep read-only historical replay, synthetic
  CPU integration, observed scientist behavior and measured GPU effects distinct.
- For rendered-input checks, compare original preprocessing versions and the
  actual supervised arrays, not just retained post-repair scripts. Moving a
  terminator from renderer to raw data may leave the intended training sequence
  unchanged. Already-rendered prompts, separate versus joint tokenization,
  masked demonstrations, explicit template tails and padded batch width are
  distinct cases. Certify a prepared artifact and its supported consumer
  separately; neither a summary JSON nor a loader call proves actual model use.
- A foreground observer's file lock can disappear while its detached child
  survives an abrupt observer death. Keep unique launch/process/exit records,
  resolve PID birth identity rather than names/PID existence, and do not turn
  missing final evidence into an automatic retry. Test actual interruption
  and observer-death behavior with owned synthetic processes.
- Match integrity checks to the actual immutable boundary. V2 pins plan
  sections0–4, not legitimate result/conclusion updates after observation.
  A whole-card byte-change gate can falsely reject correct result recording.
  Preserve before/after card hashes, verify plan/identity/lock/source separately,
  and keep a real child exit distinct from later evidence-validation failure.
  Relative input paths also need their resolution context; a v2 lock without
  caller cwd cannot silently certify that a later command reads the same file.
- Freeze source, interfaces, full component differences and the outcome vector
  before discovery. Small samples can resolve a mechanism or expose a useful
  branch; they do not identify every component's causal effect or establish
  a small stable quality gain. Promotion still needs predeclared tolerances
  and untouched held-out confirmation.
- Exercise installed native parameter/output/tokenizer types as well as inert
  stand-ins. An ordinary Enum or generic token sequence can expose a false block
  that a SimpleNamespace fixture misses. Keep native-type CPU integration,
  intercepted inference and actual model execution labelled separately.
- A superseding evidence path must not offer an unrecorded way to weaken frozen
  semantic checks. Keep the existing stop/marker thresholds or stricter values;
  legitimate exceptions remain reasoned overrides/unverified coverage, not a
  receipt made green by permissive user-controlled tolerances.
- Distinguish callback completion from a valid answer: all callbacks can return
  JSON while one explicitly returns a null/nonfinite-answer status. Actual raw
  request/draw counts, parser errors and official scores are different metrics.
- Cold CLI/tokenizer startup, preparation/full verification, steady preflight,
  loader opening and first batch costs are different phases. Report their actual
  denominators and do not extrapolate a tiny or warmed test to a full-corpus gain.
