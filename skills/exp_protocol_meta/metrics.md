# The metrics a protocol variant is judged on

All but the first come from `awm exp_protocol collect`, which reads the cards,
locks, and preflight reports in each scientist's `memory/cards/`.

| column | what it is | a good change moves it |
|---|---|---|
| `accuracy` | the official score from `metrics.json` (PostTrainBench writes `{"accuracy", "stderr"}` beside `task/`) | up, but only outside run-to-run noise (~0.01–0.03 on gsm8k) |
| `pitfalls_cost_h` | hours the scientist itself attributed to trouble before launches (`situation.pitfalls_hit[].cost_h`, summed) | **down** — this is the protocol's own KPI: GPU hours not wasted |
| `pitfalls_hit` | count of card entries above, not distinct reviewer mechanisms | down |
| `preflight_fail` | preflight failures recorded in locks (a lock is only written after preflight passes, so this counts re-runs) | down over rounds as the checks teach |
| `n_locked_open` | cards locked and never closed | down — an abandoned card is a run whose outcome was lost |
| `n_closed` / `n_cards` | how much of the work was recorded end to end | up |
| `adopted` | cards whose decision was adopt | context, not a target |
| `fields_filled` | share of required fields non-empty, averaged over cards | up; a field that stays empty across variants is either unanswerable or badly asked |
| `n_relocked` | cards locked more than once (`--relock`), not the number of relock events | inspect reasons; avoidable churn should fall, but a relock that correctly pins a repair is useful |
| `n_overrides` | pre-flight checks let through with `--override`, current and re-lock history | a check overridden in many cells is a check that is wrong for that data: fix the check |
| `n_unreadable` | `exp-NN.yaml` files that could not be parsed | zero; a non-zero count is a card written under time pressure and lost |

`session` is the directory name, except that a PostTrainBench session dir is
always `task`, so it is labelled `<cell>/task` with the parent's name.

## From the trace review

Optional lifecycle variants may add verified-outcome columns while preserving
the old conclusion-based counters. Do not reinterpret raw `n_closed`,
`n_locked_open` or `adopted` as proof that a new validation step succeeded.
Use the frozen candidate-capable reader on that candidate's bundles, check its
import/source identity, and report raw, verified, failed-closed and unresolved
outcomes separately. A portable historical receipt is not a fresh reading of
an omitted artifact. K's [prelaunch validation](../../doc/exp_protocol_iterations/2026-09-03-round-02-k-prelaunch.md)
demonstrates these consumer/measurement boundaries with synthetic fixtures;
it is not evidence of a scientist or score improvement.

`tools/exp_protocol_cell_read.py` and `tools/exp_protocol_trace_timeline.py`
read what the cards cannot say; the reviewer reports carry them per cell.

| field | what it is | a good change moves it |
|---|---|---|
| `hours_used` | wall time of the scientist session (`time_taken.txt`, or first to last turn) | context; both arms stop 1–2 h early |
| `hours_to_first_train_launch` | from the first turn to the first real training launch | down — Round 00 protocol cells took longer to launch than controls |
| `protocol_hours` | tool time in `awm exp_protocol`, card edits, reading the skill | down without `fields_filled` dropping |
| `waiting_hours` | raw tool time in sleep/tail/pgrep loops, sometimes including foreground evaluation | context only; not post-exit idle and not comparable until composite calls are separated |
| `post_exit_idle_h` | cumulative non-overlapping time per cell after the producing process exits and before the next useful action, excluding overlapping productive work | down against the predeclared per-cell threshold; report uncertain timestamp bounds and per-event maxima separately |
| `lock_before_launch` | card-matched training launches after their lock, from trace timestamps | up to all matched/matched; not an exhaustive command audit, and `collect` cannot see ordering |
| `greedy_shipped` | a greedy or measured `generation_config` written for `final_model` | verify the measured choice; Round 00's large gains are historical observations, not a universal benefit for every artifact |
| `rl_used` / `rft_tried` | on-policy RL or rejection sampling attempted, and the card's verdict | context, not a target: the protocol does not say what to run |
| `largest_eval_n` | the biggest evaluation behind a shipping decision | up to ≥500; rankings at 150–300 inverted at 500–1319 |
| `stop_reason` | the scientist's stated reason for ending, quoted | read it; an early stop with a run alive is a lost cell |

## Reading them together

Measurement checks from window 03:

- Prefer the actual inspect sample count/resolved evaluation record for `n`.
  Log bytes / 44 KB is only a rough hint. For these GSM8K binary-accuracy
  reports with the verified sample-SE convention, `p(1-p)/SE² + 1` can
  cross-check n; do not generalize that formula to other estimators or treat
  rounded scores as exact counts.
  A blocking check that promises actual-count verification must not silently
  promote this diagnostic inversion to proof, even if it fits every historical
  example. Require count provenance and handle degenerate/other estimators;
  the [Window04 decision](../../doc/exp_protocol_iterations/2026-09-04-round-02-window04-decision.md)
  rejects that proposed shortcut.
- Split composite calls before attributing time: `lock; evaluate` is not all
  paperwork, and `launch; sleep` is not all launch overhead. A mention of
  `final_model` in a card or task title is not an artifact write.
- Read retained training logs before declaring early loss unobservable.
  Distinguish terminal logged loss from whole-run `train_loss`, and record
  the logging cadence; missing individual steps are unknown, not guessed.
- For an early-stop rule, distinguish the optimizer step a line describes
  from when that line became observable. Buffered output may reveal step20
  only halfway through a run. Savings after an ideal step20 stop are not
  demonstrated savings under the emitted logs; report both timing bounds.
  Retrospective outcomes are not prospective decision inputs, and do not
  silently add exact tolerances or exclusions absent from frozen guidance.
  See the [P1 observability audit](../../doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/p1-observability-audit.md).
- A counterexample must satisfy the candidate's full frozen predicate,
  including sampled/trained parent identity and data composition. A mixed
  teacher/self stage is not interchangeable with a self-only stage.
- Do not double-count an invalid-save run's post-exit waiting under both D
  and E when reporting total savings. Low GPU memory or an unchanged log
  alone is not proof that the producing process exited.
- Fatal traceback plus GPU release can make prompt producer exit plausible,
  but a strict post-exit lower bound needs an actual exit upper bound (for
  example a timestamped producer-absence check or retained exit result).
  Later absence cannot retroactively date exit. Distinguish unconditional
  intervals from conditional teardown assumptions; a corrected premise must
  propagate to candidate retention/saturation gates, not just a footnote.
  [E2's retention re-audit](../../doc/exp_protocol_iterations/trace-reviews/window04-local/e2-retention-exit-evidence-audit.md)
  reopens an at-most6/8 proof without asserting the opposite conclusion.
- A sum of first-to-last low-memory sample spans discards singleton samples;
  report its formula/counts, not a continuous idle duration. Multiplying
  samples by the nominal interval is not a proven replacement. Exclude
  productive CPU filtering, saves and script preparation from idle, and do
  not apply a whole-cell screen threshold to selected-event coverage.
  The [control timing audit](../../doc/exp_protocol_iterations/trace-reviews/window04-local/control-timing-audit.md)
  supplies bounded examples and explains why retained copy-time mtimes are
  not producer timestamps.
- `pitfalls_cost_h` is a sum of card entries, not unique failure events. A
  follow-up card may repeat its predecessor's crash cost. Preserve the raw
  collect value; derive a separate event-deduplicated ledger with trace
  intervals for failed compute, post-exit idle and repair. Removing duplicate
  entries does not make the scientist's rounded estimate an exact duration.
- A successful lock's `preflight_fail: 0` is not a history of all earlier
  check attempts. Inspect failed checks in the trace and audit evaluation
  launch ordering separately from the existing training-launch metric.
  Before calling a violation unavoidable, check the alternatives actually
  available to the scientist, including documented reasoned overrides.
  See the [g01s07 ordering/cost audit](../../doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/g01s07-ordering-cost-audit.md).
- A scientist agreeing with a failure does not prove a runtime defect. For
  raw-field checks, inspect the original renderer/label path before the repair
  and compare the resulting model input: a data/schema rewrite may preserve
  the already-correct training sequence. The [Window04 semantic audit](../../doc/exp_protocol_iterations/trace-reviews/window04-local/card-semantics-audit.md)
  rejects a claimed65-point checker benefit because EOT was already supervised.
- Exercise a check on operations that do not use the rejected property, not
  only on stock valid/invalid inputs. A save-time hazard can apply to a merge
  but not a pure evaluation; an in-memory repair can leave the parent file
  unchanged. [Frozen D's CPU scope replay](../../doc/exp_protocol_iterations/trace-reviews/window04-local/d-scope/report.md)
  demonstrates why family labels and stock-only false-positive tests are
  insufficient. Keep counterfactual replays distinct from scientist outcomes.
- Check the launch counter's denominator before asserting universal compliance:
  the cell-reader matches the first training command for a card's script/output
  and omits unmatched launches. Audit GPU smoke/probes, retries and evaluations
  separately; a later `smoke_runs` entry is not a pre-launch lock, and a small
  run's label does not override the user's required ordering. Preserve the
  original matched ratio rather than silently expanding its denominator.
  See the [two-cell launch-scope audit](../../doc/exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/launch-scope-audit.md).
- Keep exact cohort IDs in denominators. Window 03's three new guard cells
  include two old identities and only one of the prescribed strict eight.
  An explicit tie-break after a null paired test is not a proven gain, but
  choosing a tied artifact is not itself statistical misconduct.
- Compare the observed baseline with the screen's actual acceptance threshold,
  not with perfection:7/8 already exceeds a3/4 pass rule. A passing screen
  then does not establish incremental movement; redesign before launch or
  withdraw the whole unstarted block rather than select a new success metric
  after seeing candidate outcomes.
- Use current receipt-aware result discovery to resolve eligibility. Legacy
  status files can omit newer eligibility/quarantine fields; missing is not
  false, nor permission to assume true. The strict-guard round record documents
  p00r05, whose old status-only filter incorrectly reduced the baseline pool
  from14 to13 despite current discovery validating all14.
- Identify an arm from its frozen manifest/setup and installed guidance, not
  the AWM code SHA alone or the number of cards observed afterward. A control
  can have AWM bootstrap provenance while excluding the protocol skill and
  its setup flag; Window04's c01s06 is such a protocol-free control.
- Calibrate repeat variability on the actual compared artifact. A different
  checkpoint's wider spread, even in the same cell, is not the final model's
  noise distribution. A reused `final_model` path needs its timestamp/lineage
  or content identity; equal scalar accuracy need not mean equal correct items.
- For paired outcomes, parse structured sample records and join typed item ID
  plus epoch; do not regex-stream nested JSON or zip completion-order arrays.
  Check uniqueness, selected/scored/completed counts, pair-table totals and
  numerator differences against scalar accuracy, then verify aligned inputs
  and targets. A claimed one-row omission can hide much larger pairing errors:
  [Window04 control-b](../../doc/exp_protocol_iterations/trace-reviews/window04-local/paired-counts/control-b-structural-pairs-report.md)
  corrected219/211 to58/51 with no missing samples. Dataset population size is
  not limited-run n; shared prefixes use declared dataset IDs, not returned
  completion order. An exact null paired test still does not prove equivalence.
  The [g01r03 prefix audit](../../doc/exp_protocol_iterations/trace-reviews/window04-local/g01r03-prefix-audit.md)
  shows the same error in a scientist card: its claimed150-item comparator
  took the stored array's first150 (126correct), of which122IDs were outside
  the intended dataset prefix (128correct). A declared n alone cannot prove
  subset identity or bind the card value to its referenced scalar artifact.
- Separate logged/requested knobs from resolved engine state and library/image
  identity. Equal code/template hashes or aggregate token counts do not prove
  every request or environment matched. Compare retained per-item inputs when
  available; a configuration contrast that changes both concurrency and memory
  cannot isolate either, and cross-model gaps are not a dose-response test.
- Missing from a git bundle is not necessarily lost. Check `status.skipped` and
  the original receipt-backed result directory before requesting more GPU
  reads. Conversely, logs written into ephemeral evaluator/source scratch need
  durable preservation before cleanup; a larger harvest cap alone cannot save
  them. The [P5 adjudication](../../doc/exp_protocol_iterations/trace-reviews/p5-serving-audit/planner-decision.md)
  recovered full developer metadata while leaving official per-item evidence
  explicitly unresolved. Do not bypass host trust checks to fill that gap.

- `accuracy` flat, `pitfalls_cost_h` down: the protocol saved time that the
  scientist did not convert into score. Look at what it did with the hours.
- `accuracy` up, `fields_filled` down: scientists skipped the card to go faster.
  The protocol is too expensive per card; simplify before celebrating.
- `n_locked_open` up: runs are being started and not closed. Either the close
  step is too heavy or the deadline is being hit; check `situation.remaining_h`
  on the open cards.

A loadable artifact, a clean judge, and a successful n=500 developer evaluation
do not prove official full-evaluation completion. p00r16/job 90490 passed those
stages but all nine full evaluations aborted in the numeric scorer, leaving no
`metrics.json`. Keep such attempts failed/incomplete; do not substitute the
developer score or reconstruct accuracy from a partial official run. Repeated
deterministic scorer errors need a separately scoped recovery decision, not
unbounded retries or a silent change to the frozen evaluation contract. Evidence:
[`p00r16` failure review](../../doc/exp_protocol_iterations/2026-09-03-p00r16-scorer-failure.md).

A whole-block administrative withdrawal before any job starts still needs its
receipt cancellation and terminal status harvested, but supplies neither a
scientist failure trajectory nor a validator-clean result. Keep accuracy null
and disclose the withdrawn planned cells separately; absence of judge flags
does not make an unstarted job a clean observation. The strict-guard round's
C withdrawal demonstrates this distinction.
