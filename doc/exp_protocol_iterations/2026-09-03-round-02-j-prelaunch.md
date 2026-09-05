# Round02 J — 2026-09-03 prelaunch construction

Status: **0/4 run; no receipt, release or promotion**. The candidate is frozen by this record's introducing commit. Its resolved SHA/tree and immutable manifest are recorded in the [J spec](../spec/2026-09-03-exp-protocol-round02-j-lock-scope.md) after construction.

## Variants

| label | commit / tree | difference |
|---|---|---|
| guard drift baseline | `2f64581` / `189319d6` | unchanged guard |
| J lock scope | this record's introducing commit | rule1 covers short training/evaluation runs; template smoke comment no longer implies exemption |

Operator parent `1a141a1` matches the drift baseline on all six shipped paths. J is not stacked on E v2, C, H or a prospective-comparator change.

## Cells

Four independent `j02r01–04`, run_index1; GSM8K, Gemma-3-4B at the same pinned revision, scientist Opus5[1m] high/1M,10h each. PTB, images, official evaluation/judge contract and six shipped paths remain the Round02 contract. Held-out task: none.

## Results

No J scientist has run; no collect CSV, J cards, accuracy or compliance pass rate exists yet.

| variant | accuracy mean (min–max) | pitfalls_cost_h sum | n_locked_open sum | fields_filled mean |
|---|---|---|---|---|
| J,0/4 | unmeasured | unmeasured | unmeasured | unmeasured |
| matched drift | not yet run | unmeasured | unmeasured | unmeasured |

The three prior guard cards read by the planner are listed in the [strict Round01 record](2026-09-03-round-01-strict-guard.md): g01s03 exp-05, g01s08 exp-02 and g01s04 exp-03. J additionally uses targeted raw g01s01/g01s07 launch traces, lock history and card declarations in the [scope audit](trace-reviews/round01-strict-guard-addendum/launch-scope-audit.md). Three J cards and matched-drift cards must be reviewed after they exist; that postlaunch requirement is not marked complete now.

## Trace review

The completed [strict-cohort planner decision](trace-reviews/round01-strict-guard-addendum/planner-decision.md), full local Opus max synthesis and additional audits distinguish card-matched training counts from all training/evaluation execution. g01s01's training smoke at10:01:24 precedes the10:21:01 lock; g01s07's10:04:01 smoke precedes10:20:05. Both were actual GPU runs, not CPU static checks. Separate pre-lock evaluation findings reinforce the scope question without changing the guard's predeclared session-end-harm outcome.

## Directions

Ledger #26 becomes J. #27 future-comparator semantics remains separate. C v2's four unstarted jobs have been cancelled and harvested;29 held jobs remain. D/B/H retain first scientific priority, A/E v2 follow with drift pairs. J is prepared for a later independent screen, not silently inserted into their frozen trees.

## Decision

Build the independent screen and restore guard source after freezing; **no promotion or launch**. Primary: zero training/evaluation launches without a matching successful pre-launch lock in all four evaluable cells, using a complete command inventory. Include short smokes, probes, retries and evaluation-only launches; unknown/unmatched scope is not a pass, and no exposure is uninformative. The old matched-training counter is secondary, not the denominator for this claim.

Frozen score guardrail: the24-cell protocol reference pool specified in the J spec has mean0.7037212534748547; block mean must be at least **0.6737212534748547**. This is a coarse historical guardrail, not causal evidence. Also inspect card burden, diagnostic avoidance, fabricated fields and first-action/session-end compliance. A winner needs another independently frozen four-cell block and eventually held-out confirmation.

## Change

Only rule1 and the template's smoke-record comment change. Short training/evaluation smokes and probes require a matching card and successful prelaunch lock; an unrelated earlier card, an empty slot, a later smoke entry or a failed pipeline lock is not coverage. An exact dependent evaluation already declared in a locked training card uses that card. CPU-only static preparation that does not train/evaluate remains possible before the card. Existing material-change/sweep and reasoned-override rules are preserved.

No schema field, hook logic, preflight check, comparator semantics, training recipe, model or budget changes. The template's parsed YAML is identical; only comments differ.

## Evidence and validation

- `pytest -o addopts='' -q tests/test_exp_protocol_hook.py tests/test_exp_protocol_skill_files.py tests/test_exp_protocol_install.py tests/test_exp_protocol_lock.py`: **34 passed**.
- Parsed-template equality, SKILL prefix/suffix equality around rule1, and all-six-shipped-path diff prove confinement to the two declared text surfaces. `git diff --check` passes.
- Independent read-only reviewer `/root/j_forward_check` read only the skill, template and existing example, not this spec or the source audits. It answered all six scenarios correctly: new training smoke waits for matching lock; failed lock hidden by pipeline exit0 still blocks; tokenizer/AST/data-only preparation proceeds; already-declared dependent evaluation reuses its card; two sweep configurations use separate cards; model-backed generation **with correctness grading** is evaluation and cannot be authorized solely by `setup.data[].build_command` or an unlocked future training card.
- The reviewer suggested more examples/success-check mechanics. Those did not change its decisions; no extra example, synthetic prior-lock provenance or schema was added. Pure generation without grading was noted as a separate unresolved boundary, not silently added to or exempted by this training/evaluation intervention. This is scenario review, not an end-to-end scientist run.
- Generic skill validation still rejects the pre-existing underscore name `exp_protocol`; preserve runtime compatibility rather than rename it. Repository schema/install tests pass. Manifest checks follow once the commit/tree exist and do not override ownership or isolation gates.

## Next round

Freeze the manifest, restore the guard tree and verify structural and cross-variant consistency. No J receipt while OWNERSHIP FAIL persists. Continue the hourly monitor and terminal harvesting; design #27 independently. Only after operational gates and complete launch-scope review may a held J receipt be considered for release. No running job is cancelled by this preparation.
