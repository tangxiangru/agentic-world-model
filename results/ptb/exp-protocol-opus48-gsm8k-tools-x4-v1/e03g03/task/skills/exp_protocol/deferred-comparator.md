# When this card will produce its comparator

Use this only for a within-card comparison whose comparator output is not yet
available. Ordinary already-measured comparators keep the normal workflow.

Before either arm runs, complete the card and declare:

```yaml
hypothesis:
  expected_effect:
    metric: accuracy
evaluation:
  protocol:
    n: 500
  comparator:
    ref: exp-02
    value: null
    path: /home/ben/task/eval/exp-05-parent.json
    defer_validation: true
```

These are only the relevant fields, not a complete card. The command that
produces both measurements must be declared as usual. Run check and lock
before launching it. A missing planned file yields a deferred warning; an
existing invalid report still fails. The mode is recorded in the lock and
cannot be removed by re-locking the same card. Use a new card for that change.
Use a distinct per-card output path; overwriting an earlier card's comparator
would invalidate that earlier receipt.

Keep sections0–4 unchanged, including the null value and true flag. Record
observations in sections5–6, not as a guessed prelaunch comparator value.

## Evidence needed at close

The planned JSON path must hold the actual completed evaluator result. A
minimal summary can contain `{"n": 500, "accuracy": 0.72}` when those values
come from the actual completed evaluation—not from the requested limit or
the desired outcome. Preserve the source evaluation log for audit. A raw
successful Inspect JSON report is also accepted: evaluated, completed and
scored counts must agree with the locked n, and the named metric must be finite.

The input dataset population and requested limit are not actual evaluated n.
Neither file size nor an accuracy/standard-error inversion supplies missing n.
If a PTB summary contains only accuracy/stderr, retain the full evaluator report
or derive a summary from its actual completed metadata. Do not edit the grader
or invent fields. Unknown, partial, malformed or mismatched evidence cannot
close a completed opted-in experiment. Dev set, seed, decoder and model identity
still need the same manual verification as rule2; the machine does not prove
those facts merely by matching n.

Record at least one target measurement for the named metric at the locked n,
with its finite value and source path. Other-n probes are separate observations,
not this comparison. The receipt verifies comparator-file evidence and this
recorded target-count consistency; it does not independently audit the target
file or prove complete experimental correctness.

After filling results, run `awm exp_protocol close --dir <dir> <card>` and
check that it succeeds. It writes `exp-NN.comparator.json`, binding the card,
plan, lock and verified evidence. Index/starting points, collection and the
Stop hook distinguish an unverified conclusion from this receipt. Keep the
receipt and source evidence; changing a card or available evidence invalidates
the receipt and requires valid closure again. Do not forge or hand-edit it.

`collect` preserves raw `n_closed`, `n_locked_open` and `adopted` for comparison
with older variants. For this mode read `n_deferred_verified`,
`n_deferred_failed_closed` and `n_deferred_unverified`; a filled conclusion can
increase raw `n_closed` while remaining unverified and unavailable as a starting
point. On a relocated/harvested copy, a matching receipt can attest historical
verification without the original raw file. Re-closing is different: it needs
the original declared evidence accessible again and must not invent a new
verification merely because a historical receipt exists.

## If the paired experiment fails

Record the failed/killed execution and its reason, an inconclusive verdict
and a non-adopt decision; an unrun experiment needs a reasoned summary.
Partial observations can remain recorded, but they do not certify a completed
comparison. Close then writes a distinct failed/unrun receipt without a
verified metric. A supported/contradicted/adopted result cannot use this route.

The Stop hook's12-block cap is unchanged. If the CLI is broken, record the
failure honestly; manually filling conclusion is not verification for this
mode, and the card must not be used as a validated starting point. The receipt
is reproducibility evidence, not protection against arbitrary filesystem edits.
