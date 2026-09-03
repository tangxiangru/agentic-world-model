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
