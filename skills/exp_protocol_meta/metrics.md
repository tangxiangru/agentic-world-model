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

Not in `collect` yet, worth adding when the trajectory conversion is wired in:
wall-clock spent training vs idle (from `awm traj spans`), and the number of
distinct experiments per hour.

## Reading them together

- `accuracy` flat, `pitfalls_cost_h` down: the protocol saved time that the
  scientist did not convert into score. Look at what it did with the hours.
- `accuracy` up, `fields_filled` down: scientists skipped the card to go faster.
  The protocol is too expensive per card; simplify before celebrating.
- `n_locked_open` up: runs are being started and not closed. Either the close
  step is too heavy or the deadline is being hit; check `situation.remaining_h`
  on the open cards.
