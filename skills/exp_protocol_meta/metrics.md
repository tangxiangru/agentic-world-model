# The metrics a protocol variant is judged on

All but the first come from `awm exp_protocol collect`, which reads the cards,
locks, and preflight reports in each scientist's `memory/cards/`.

| column | what it is | a good change moves it |
|---|---|---|
| `accuracy` | the official score from `metrics.json` (PostTrainBench writes `{"accuracy", "stderr"}` beside `task/`) | up, but only outside run-to-run noise (~0.01–0.03 on gsm8k) |
| `pitfalls_cost_h` | hours the scientist itself attributed to trouble before launches (`situation.pitfalls_hit[].cost_h`, summed) | **down** — this is the protocol's own KPI: GPU hours not wasted |
| `pitfalls_hit` | count of the above | down |
| `preflight_fail` | preflight failures recorded in locks (a lock is only written after preflight passes, so this counts re-runs) | down over rounds as the checks teach |
| `n_locked_open` | cards locked and never closed | down — an abandoned card is a run whose outcome was lost |
| `n_closed` / `n_cards` | how much of the work was recorded end to end | up |
| `adopted` | cards whose decision was adopt | context, not a target |
| `fields_filled` | share of required fields non-empty, averaged over cards | up; a field that stays empty across variants is either unanswerable or badly asked |
| `n_relocked` | cards locked more than once (`--relock`) | down; each one has a reason in its lock file — read them |
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
| `waiting_hours` | tool time in sleep/tail/pgrep loops on running jobs | context; it is the training time seen from the trace |
| `lock_before_launch` | training launches after their card's lock, from trace timestamps | up to all/all; `collect` cannot see ordering |
| `greedy_shipped` | a greedy or measured `generation_config` written for `final_model` | up — 4/9 protocol vs 7/7 control in Round 00, worth 7–16 points per cell |
| `rl_used` / `rft_tried` | on-policy RL or rejection sampling attempted, and the card's verdict | context, not a target: the protocol does not say what to run |
| `largest_eval_n` | the biggest evaluation behind a shipping decision | up to ≥500; rankings at 150–300 inverted at 500–1319 |
| `stop_reason` | the scientist's stated reason for ending, quoted | read it; an early stop with a run alive is a lost cell |

## Reading them together

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
