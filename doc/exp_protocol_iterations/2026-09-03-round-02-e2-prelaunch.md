# Round 02 E v2 — 2026-09-03 prelaunch construction

Status: candidate frozen by the commit that introduces this record; **0/4 run, no receipt, no release, no promotion**. The resolved commit/tree and immutable manifest are recorded in the [E v2 spec](../spec/2026-09-03-exp-protocol-round02-e2-process-wait.md) after construction. This is a prelaunch record, not a completed round.

**2026-09-04 correction:** the historical0.433/0.234 h post-exit lower bounds below are conditional on producer exit by GPU release, not unconditional measurements. The [focused audit](trace-reviews/window04-local/e2-retention-exit-evidence-audit.md) reopens the strict non-saturation proof without claiming saturation or changing the frozen intervention. E2 remains unregistered pending adjudication; CPU validation remains valid but does not establish the scientific retention gate.

## Variants

| label | commit of skills/exp_protocol | what differs from baseline |
|---|---|---|
| guard drift comparator | `2f64581`, protocol tree `189319d6` | unchanged guard source and same six shipped infrastructure paths |
| E v2 | this record's introducing commit | one coordinated process-waiting intervention in rule 9, stop-hook reason and lifecycle-pitfall guidance/source |

The operator parent `b1cdd19` matches `2f64581` on all six shipped paths. E v2 is built from that guard baseline, not stacked on A/B/C/D/H or the old E tree `58af0780`.

## Cells

Task: GSM8K · base: `google/gemma-3-4b-pt` at `cc012e0a6d0787b4adcc0fa2c4da74402494554d` · scientist: `claude-opus-5[1m]`, high, 1M context · budget: 10 h per cell · four independent replicates `e02s01–04`, run index 2. Held-out task this round: none; AIME is not used for iteration.

PTB, images, evaluation/judge profiles and the six shipped paths stay identical to the original E screen and the matched guard drift setup. The replacement manifest will freeze only the new candidate identity and immutable batch/cell/run identity; scientific settings do not change.

## Results

No candidate has run. There is no candidate collect CSV, score, time-saving estimate or valid pass rate yet.

| variant | accuracy mean (min–max) | pitfalls_cost_h sum | n_locked_open sum | fields_filled mean |
|---|---|---|---|---|
| E v2 (0/4) | not measured | not measured | not measured | not measured |
| matched drift | not yet run | not measured | not measured | not measured |

Three source guard cards read by the planner (not E v2 outcomes):

- `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s03/task/memory/cards/exp-05.yaml`: failed initial Trainer save is recorded separately from the successful relaunch; trace/monitor, not the card's aggregate cost, bounds the first attempt's post-exit wait.
- `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s08/task/memory/cards/exp-02.yaml`: OOM at step 100, no checkpoint or measurement; a recorded failure is not negative evidence for the scientific hypothesis.
- `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s04/task/memory/cards/exp-03.yaml`: evaluation-only producer and existing symlinked weights illustrate why completion guidance must be stage-neutral; this card does not supply an additional E failure bound.

Read three candidate cards and the matched comparator cards after results exist; that postlaunch review is not complete now.

## Trace review

Prior window: [window03 local review](2026-09-03-trace-review-window03-local.md). Incremental strict cohort reports: [g01s03](trace-reviews/round01-strict-guard-addendum/cells/g01s03.md), [g01s08](trace-reviews/round01-strict-guard-addendum/cells/g01s08.md). Local Opus max session `73c154b4-944d-4e7d-bc60-5c192ed41207` supplied the four-cell addendum; `35a6a6d1-b646-479b-ac76-8d26afcb5d0c` is still reviewing g01s02/05/07. Full strict-guard safety synthesis remains open.

The two conservative single-event post-exit lower bounds are 0.433 h and 0.234 h, each above the frozen cumulative 0.15 h/cell threshold. Thus at most 6/8 of the exact strict cohort can pass, independently of the remaining reviews. This retains E but neither establishes a score effect nor passes the full guard safety gate.

## Directions

Ledger #15: retain E and prepare its new immutable revision. Do not take the G/P1 substitution branch. Other screens remain independent and frozen; D/B/C plus drift A retain first-wave scientific priority. Old E jobs 91064–91067 must remain held until a valid replacement is registered and the whole unstarted old block can be withdrawn safely.

## Decision

Keep the branch's guard tree as the comparator after freezing E v2. The candidate is **unproven**: primary screen target is cumulative non-overlapping post-exit idle <0.15 h/cell for training, sampling and evaluation, with uncertainty bounds. Per-event maximum and proxy-only liveness decisions are secondary. False death classification, premature closure or lost work are failures, not savings. The score guardrail is the predeclared protocol-baseline pool mean minus 0.03. A winner needs a separately frozen second four-cell block before a score claim and held-out confirmation before promotion.

## Change

Prefer a foreground producer/exit-check/dependent-stage script. For background children, use same-shell `wait`; otherwise retain a task-completion handle or launch-wrapper exit result and poll the actual producer at most 60 seconds apart. A missing exit result is unknown, not success. Quiet logs, old files and residual GPU memory prove neither life nor death. Verify current-invocation artifact identity and required contents before accepting results or chaining a dependent stage.

Only three coordinated text surfaces change. Hook control flow, block limit, card schema, preflight checks, recipe, budget and model do not.

## Evidence and CPU validation

- The [retention spec](../spec/2026-09-03-exp-protocol-round02-e2-process-wait.md) records exact trace/monitor timestamps and receipt-backed job IDs 90793/90798.
- Independent read-only forward reviewer `/root/e2_forward_check` exercised four scenarios: live but quiet producer, exit 0 with an old metric artifact, background child of an earlier shell, and failed trainer with a separate engine holding GPU memory. Its artifact-attribution and cross-shell exit-recovery clarifications were incorporated in all three surfaces. This is scenario review, not an end-to-end scientist experiment.
- `pytest -q tests/test_exp_protocol_hook.py tests/test_exp_protocol_skill_files.py tests/test_exp_protocol_install.py tests/test_exp_protocol_lock.py`: **34 passed** on 2026-09-03.
- AST comparison excluding `REASON` proves hook logic unchanged; parsed YAML comparison excluding the lifecycle entry's guidance/source proves other pitfall data unchanged; rule-9 prefix/suffix comparison proves the rest of SKILL.md unchanged.
- Six-path comparison against `2f64581` differs only in those three files. `git diff --check` passes.
- Generic skill `quick_validate.py` rejects the existing underscore name `exp_protocol`; this is a pre-existing naming mismatch, not a new candidate defect. Preserve the runtime installation name; repository parsing/install tests pass.
- Manifest validation is performed after resolving the candidate commit/tree. Structural checks do not override the independent ownership/isolation release gate.

## Next round

Freeze the replacement manifest, restore the guard source tree and validate comparability. **Do not submit while OWNERSHIP FAIL is active.** After restored ownership and the required native isolation, register a held replacement receipt before withdrawing old E as a whole block. Never cancel running jobs. Continue harvesting the current wave and finish the exact eight-cell strict guard synthesis before deciding its safety gate.
