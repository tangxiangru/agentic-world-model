# HumanEval semantic decontamination guard

Status: preregistered design-only protocol/harness candidate. No current cell,
score, judge, scorer or WMA skill is changed. No job is submitted in this check.

## Trigger and boundary

HumanEval protocol-only c56r03 locked a110,689,404-byte SFT file after the
existing n-gram contamination check returned zero. The official semantic judge
later flagged contamination, reporting279 documents representing67
HumanEval-idiosyncratic function families in the Magicoder OSS/Evol lineage.
Training started on this frozen file, then the separate background-wait lifecycle
failure ended the cell before any checkpoint or final model existed.

This is a protocol/data-boundary problem, not WMA policy. The scientist-side WMA
was absent and no prediction rule can make contaminated training valid. Keep the
original contamination/general flags and incomplete status. Do not expose held-out
prompts or solutions to the scientist or WMA while implementing the guard.

## Candidate

Add a trusted prelaunch HumanEval semantic screen, outside the scientist/WMA
tool surface. It may use protected benchmark references internally, but returns
only pass/fail, counts, stable hashes and coarse categories. It must identify
canonical function/signature/docstring/doctest families and close paraphrases,
not only contiguous n-grams. Dataset lineage is evidence for scrutiny, not an
automatic blacklist: evaluate actual selected training bytes.

The implementation change belongs in protocol/harness code and must be one
candidate commit. It must not modify the WMA skill, result scorer, historical
judge outputs or current frozen manifests.

## Preregistered checks

- Positive regression: detect the already frozen c56r03 file and a separately
  frozen held-out set of semantic HumanEval variants. Report the c56r03 count
  without copying protected text into public/operator summaries.
- Negative regression: accept independently clean code-instruction and BFCL
  function-call corpora selected before this candidate, with reviewed false
  positives listed. Do not call all Magicoder rows contaminated solely by source.
- Evasion canaries: renamed functions, paraphrased docstrings, reordered doctests,
  equivalent type annotations and wrapper code around a canonical solution.
- Leak guard: scientist/WMA receives no reference prompt, target, solution,
  matching substring or item identity; only bounded aggregate diagnostics.
- Cost guard: CPU preflight wall time <=10minutes for a100MB/25k-row corpus and
  no GPU allocation; memory and output stay bounded. Failure is fail-closed.
- PTB guard: unchanged final evaluation, judges and existing n-gram output remain
  recorded alongside the new semantic result; no post-hoc score rescue.

Falsify if it misses known positive canaries, requires exporting protected text,
cannot distinguish current contaminated bytes from clean negatives, exceeds the
cost bound, or can be bypassed by the registered semantic canaries.

## Promotion and rollout

Run zero-GPU unit/canary validation first, then a validation-only frozen-source
PTB preflight. Do not retrofit it into running/pending Opus4.8 cells or selectively
replace failed results. A later scientific cohort must apply the accepted common
guard symmetrically to every HumanEval arm. Existing first-wave outcomes remain
their own runtime cohort and cannot promote this guard by themselves.

Implementation is deferred while the shared PostTrainBench submodule contains a
preserved external update and S0 validation92312 is pending against its frozen
source. Before editing, fetch/reconcile that work safely and recheck ownership.
