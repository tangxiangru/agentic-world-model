# Process checks from experiment experience

Use the section relevant to the operation being planned. These are scoped
execution/evidence practices, not instructions to choose a dataset, numerical
precision, training method, merge or research branch. Run cheap artifact/CPU
checks before expensive work; any model-training/evaluation probe requires
its own or an explicitly covering prospective checked/locked card.

## Actual data and rendered inputs

- Record the materialized input path/hash, source revision, actual rows and
  the transformations applied. When comparing data composition, distinguish
  unique questions, distinct question/solution pairs and repetition weights.
  A larger file or per-question cap does not prove more question coverage or
  a superset. Define how identities are counted; exact strings are not a
  proof of semantic equivalence.
- Verify realized prefix/shot and length distributions after filtering,
  sampling and augmentation. A configured fraction is not the actual one;
  duplicated mutable objects can receive an augmentation more than once.
  Reuse the build's audit output instead of re-tokenizing everything per card.
- Inspect the actual renderer, tokenizer and supervised labels: prompts have
  the intended special tokens once, demonstrations are masked as intended,
  and the completion retains the declared terminator. Measure rendered
  lengths, retained/truncated/dropped rows and supervised-token counts under
  the real truncation/padding policy. Preserve the check's coverage; a sample
  cannot certify every row. Respect the frozen preflight thresholds; don't
  lower them to erase a failure.
- A raw-field check can disagree with a correct runtime renderer. Compare
  the effective supervised sequence before changing the data. A reasoned
  override needs evidence of the specific false alarm; it is not permission
  to ignore a genuine mismatch. A template or source claim alone is not proof
  of the inputs actually used.

## Effective execution and numerical state

- Record requested and effective decode settings, including model-level
  defaults, relevant request overrides, token cap and stop handling. Use
  available engine output/config evidence; distinguish inference from a
  logged effective value. Writing `temperature: 0` in a card does not set it.
  Which decode policy to choose remains a measured research decision.
- Distinguish parameter/master-weight dtype, optimizer-state dtype, autocast
  and exported-weight dtype. FP32 state is not necessarily FP32 master
  weights; a BF16 export does not reveal the training update precision.
  Inspect the actual loader, optimizer path and runtime evidence.
- When diagnosing apparently flat training, inspect retained loss/gradient
  logs, logging cadence and optimizer step count. If a planned diagnostic
  measures parameter deltas or unchanged-weight fraction, record its scope
  and cost. Flat loss alone does not establish data saturation, numerical
  failure or which recipe should run next. Missing deltas remain unknown;
  do not add an expensive training run just to fill a field.
- For sampling, verify actual tokenized inputs and explicit stop behavior
  on a prospectively declared probe. Inspect finish/stop reason or token IDs
  when available: literal stop-token text may be omitted by the engine.
  Save raw draws before answer parsing/filtering; keep nonfinite/parse errors
  distinct from model mistakes and count actual requests/draws/retained rows.

## Comparisons and complementary results

- Bind each measurement to its checkpoint/config identity, evaluator,
  dataset/revision, actual scored population, effective serving settings and
  invocation. A reused directory name is not immutable model identity.
- For matched comparisons, join typed `(sample.id, epoch)` keys, checking
  duplicates/missing records and aligned inputs/targets/rendered prompts.
  Use the declared dataset IDs for a prefix, never `samples[:N]` from a
  completion-order array. Requested limit, dataset population, completed n
  and scored n are different values.
- Report paired fixes/regressions and uncertainty as well as scalar scores.
  On intentional treatment contrasts, identify the changed setting; do not
  label a changed token cap, decoder or population as pure repeat noise.
  Record the n used to choose an artifact separately from a later packaging
  check. A non-significant result does not prove equivalence.
- If complementarity is relevant to a declared comparison, report A-only,
  B-only, both-correct and both-wrong counts. An actually tested combination
  also needs its fixes and breaks in those groups. Equal scores can hide
  distinct errors, but disagreement alone does not establish stable utility.
  An oracle using gold to choose the right answer is an upper bound, not a
  deployable system; never report it as an achieved accuracy.
- Preserve selection versus independent-confirmation provenance. Reusing a
  selection set is not fresh evidence. Do not subtract counts from different
  invocations to invent an unobserved subset score. Benchmark access and
  test-data restrictions in the main skill continue to apply.

## Checkpoint lineage and export

- Before execution, name actual parents, initialization versus continuation,
  and the checkpoint retention plan. Record produced/retained identities and
  their measurement paths after execution. A rejected claim does not change
  the retention promise. You choose the storage budget, branches worth keeping
  and whether to combine models; this skill supplies evidence for that choice.
- Apply save checks to actual operations that serialize a model/config.
  Evaluation-only loading does not exercise a Trainer save. A script may
  repair its in-memory config while the parent file stays unchanged; inspect
  that path rather than reject every greedy parent or every `merge` label.
- Distinguish valid checkpoint serialization from the selected serving
  configuration. Preserve the selected serving bytes/identity across a safe
  save and confirm they reach the exported artifact. A save-path repair must
  not silently change the selected decoder.
- Verify the selected export includes weights/shards, tokenizer and required
  processor/config assets. Metadata/file presence, native loader acceptance
  and an actual evaluator result are separate evidence. CPU metadata checks
  alone cannot certify a vLLM invocation. Keep a usable incumbent while
  staging and validating a replacement; do not delete it on partial success.

## Run lifetime and recoverable output

- Use a unique invocation/output location and capture the actual producer
  identity at launch, including a start/birth identity where supported.
  Process-name searches and a later reused PID are not reliable identity.
  Retain exit status from the launcher/parent that owns the child (`wait`
  in the same shell where possible); a different shell cannot recover it
  simply by waiting on the numeric PID.
- Choose a launch lifetime that covers the work and survives bounded tool
  observation timeouts. Long foreground producers can die at the tool's
  timeout; detached producers can outlive their observers. Track both and
  keep the scientist session active while required work remains alive.
- A quiet log, existing partial output, file mtime or empty GPU process list
  is not successful producer exit. Verify this invocation's exit and required
  output contents before using them. Missing evidence remains unverified;
  investigate the existing run before starting a duplicate.
- Preserve raw/partial outputs, errors and failure records so a parser or
  save error does not erase what was learned. Separate failed compute,
  recovery work and post-exit idle; do not add the same interval twice.
  Close failed/killed experiments honestly without claiming their hypothesis
  was falsified merely because execution failed.
- `unverified` is an evidence description, not a v2 `result.execution` enum.
  If the producer is confirmed no longer alive but required completion/exit
  evidence cannot be recovered, record the experiment as `failed` with an
  explicit evidence-validation failure and `conclusion.verdict: inconclusive`;
  keep the unknown OS exit status in the linked notes. Do not invent a kill or
  successful exit. If the producer may still be alive, investigate that
  uncertainty before ending the session rather than using closure to abandon it.

## Current v2 evidence limits

These practices use existing card fields plus linked local evidence files;
they do not add an undocumented CLI, schema field or automatic verification.

- `stop_token_consistent`/`answer_marker_single` read sampled raw fields;
  `max_seq_len_headroom` uses a character estimate. None runs the renderer.
- `setup.data` is still a nonempty list for every family in this installed
  schema, with positive `n_examples`. For non-training cards, use genuinely
  applicable input data and describe its role. Do not fabricate a data file
  or count to satisfy it. If there is no representable real input, record the
  schema limitation; an override of preflight does not bypass schema errors.
- No automated deferred-comparator contract is supplied here. For an
  evaluation card that will produce both measurements, keep unavailable
  comparator value/path null, describe both reads and their output paths in
  the locked plan/diagnostic, then place actual values and paths in `result`.
  A command can be a declared script running those reads. Do not modify the
  locked plan with the observed answer, invent a placeholder result, or run
  the evaluation before lock. Validate both outputs before drawing the
  comparison; CLI acceptance alone does not verify them.
- The comparator check can print PASS when n is unverifiable, and `close`
  checks result shape and lock integrity rather than all measured files,
  sample identities or model loadability. Report actual verification and
  unknowns separately. Do not claim unimplemented automation has run.
- `awm exp_protocol new` creates a minimal card; it does not copy every
  optional template field. Add relevant existing fields explicitly before
  lock. New observations go in `result`, since all plan fields are frozen.

## Where to record evidence

Use `situation.trigger_evidence`/`problem.evidence` for observations already
available before launch, `setup.data[].selection` and method descriptions for
declared settings, and `evaluation.diagnostic` for planned measurements.
Afterward, use `result.measurements`, `training_summary.notes`,
`diagnostic_result` and `checkpoints_kept` to reference real local records.
Do not invent values or evidence paths. Record a newly noticed opportunity
and its uncertainty in the conclusion; it does not retroactively prove the
original hypothesis or dictate the next experiment.
