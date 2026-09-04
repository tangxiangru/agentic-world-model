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
- Use independent forward tests on realistic inputs, not only regressions
  that encode the intended answer. Keep read-only historical replay, synthetic
  CPU integration, observed scientist behavior and measured GPU effects distinct.
- Freeze source, interfaces, full component differences and the outcome vector
  before discovery. Small samples can resolve a mechanism or expose a useful
  branch; they do not identify every component's causal effect or establish
  a small stable quality gain. Promotion still needs predeclared tolerances
  and untouched held-out confirmation.
