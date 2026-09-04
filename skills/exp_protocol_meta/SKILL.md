---
name: exp_protocol_meta
description: Use when iterating the experiment protocol itself on the GPU cluster — choosing protocol variants, launching a batch of scientist cells per variant, collecting the numbers, deciding one change, and recording why. Read by the iteration agent only; never installed into a scientist's task directory.
---

# Iterating the experiment protocol

You are not the scientist. You run scientists — several at once, on different
versions of `skills/exp_protocol/` — and change the protocol based on what the
cards and the scores say. The scientist never sees this file.

## What you may change

| in `skills/exp_protocol/` | how |
|---|---|
| `SKILL.md` wording and rules | edit; one rule per round if you can |
| `pitfalls.yaml` entries | add with a `source`; promote `check: null` to a check id once a check exists |
| preflight checks (`awm/exp_protocol/preflight.py`) | add a `@check` with a test in `tests/test_exp_protocol_preflight.py` |
| `card.template.yaml` fields | adding an optional field is fine; changing a required one bumps `schema_version` to v3 with a note in `doc/spec/` |

You do not change `exp_protocol_meta` from inside a round. Changes to the loop
itself are a separate, human decision (this revision — the trace review by
subagents, parallel single-item candidates, 4-cell screens — is the user's
directive of 2026-09-03, recorded in the iteration-basis spec §七 and in
`doc/exp_protocol_iterations/directions-ledger.md`).

## The loop

1. **Pick the variants.** A variant is a commit of `skills/exp_protocol/` (plus
   the preflight code it relies on). Name each by its short sha. One of them is
   always the current baseline; up to three single-item candidates may run
   beside it in one wave, each exactly one item away from the baseline tree
   (never stacked: on a linear branch, revert the previous candidate's change
   before adding the next, and freeze each by `awm.sha` + `awm.protocol_tree`).
2. **Fix everything else.** Same task, same base model, same scientist model and
   effort, same `PTB_NUM_HOURS`, across variants. The protocol is the only
   thing that differs between cells.
3. **Seeds.** Two cells per variant is the floor below which a variant does not
   count; the working sizes are a 4-cell screening block per candidate
   (accuracy is a guardrail there — n=4 resolves about 0.06 — and the candidate
   is read on the metric it was built to move) and a second 4-cell block for
   a winner before any score claim. Run-to-run variance on PostTrainBench is
   large: one cell per variant decides nothing, and the baseline pool keeps
   growing by two cells per wave.
4. **Held-out task.** One task is never used for iteration; it is run only to
   confirm a change generalises before it becomes the baseline.
5. **Launch.** Each scientist cell in this line uses the `claude_vertex_high_awm`
   scaffold and declares, in its manifest cell, an `awm` block: the commit to
   ship (`sha`; it must carry `awm/sandbox.py`, so it is the branch head, not
   the commit that last touched the skill), `protocol_tree` =
   `git rev-parse <sha>:skills/exp_protocol` (the variant's identity: two
   commits with the same tree are the same variant, and `awm ptb check`
   refuses a cell whose tree is not the one declared), `paths` = the six
   entries of `EXP_PROTOCOL_SHIP` in `awm/ptb_experiments.py` (the CLI,
   `awm/exp_protocol`, `skills/exp_protocol` and nothing else), and
   `setup: "--exp-protocol --tool claude"`. The launcher ships exactly those
   paths of that commit into the sandbox, read-only, and the scaffold runs
   `awm sandbox setup` before the prompt.
   The meta skill, the docs, and anything of the world-model agent
   (`awm/wma`, `skills/wma*`) can never be shipped: the launcher refuses
   them by name, and refuses `awm` or `skills` wholesale. Use the committed batch launcher (`awm ptb`, manifests
   under `experiments/posttrainbench/`) and the queue file the operator
   reconciles (`doc/reference/ptb_operator_runbook.md`); do not hand-craft
   sbatch files or run `sbatch` yourself.
6. **Collect.** Results come back as bundles the operator commits under
   `results/ptb/<batch>/<cell>/`:
   ```bash
   awm exp_protocol collect results/ptb/<batch>/*/task --csv > round-NN.csv
   ```
   `metrics.md` defines each column and what a move in it means.
7. **Analyse.** Per variant: mean and range of `accuracy`; sum of
   `pitfalls_cost_h`; `n_locked_open` (cards started and abandoned);
   `fields_filled`; `n_overrides` (a check overridden in many cells is a
   check to fix, not a scientist to blame). Then run the **trace review**
   (`trace_review.md`): reviewer subagents read every clean cell's trace in
   groups of three or four with a fixed brief and write one report per cell
   (`tools/exp_protocol_cell_read.py`, `tools/exp_protocol_trace_timeline.py`
   give them the facts and the hours); a synthesis subagent ranks the
   explanations of the score difference and proposes candidates. This is a
   deep review, not a score-table checkpoint: add reviewer subagents whenever
   a window is too large or heterogeneous for the fixed groups, and do not
   stop until every clean trace is covered. Read the synthesis and three cards
   per variant yourself — the numbers say whether, the traces say why.
8. **Decide the candidates.** Each candidate is one item, traceable to at
   least two cells of the review: a pitfall that cost hours and has no entry;
   a rule scientists skipped; a field that stayed empty; a decision the other
   arm made and this arm did not. Declare the metric the 4-cell screen will
   read for it and the score guardrail. Write it up first (spec, ledger), then
   make it. At the same decision point, audit every held block. Withdraw an
   entire not-yet-started block when the new evidence makes it scientifically
   unnecessary; never select individual cells by outcome, never cancel a
   running job, and never cancel outside the immutable receipt's job IDs.
   Replenish before the valid `PENDING(JobHeldUser)` floor falls below eight.
   A losing candidate's pending cells are withdrawn; a winner earns its second
   block.
9. **Record.** Copy `iteration_record.template.md` to
   `doc/exp_protocol_iterations/<date>-round-NN.md`, fill every section, commit
   it with the change in the same commit; keep
   `doc/exp_protocol_iterations/directions-ledger.md` current — every direction
   with its status and what would change it, every decision with its
   alternatives. Do not leave reusable knowledge only in chat, Claude logs, or
   a one-off synthesis: durable loop/process lessons go into this meta skill or
   its linked `trace_review.md`/`metrics.md`; experiment-specific findings and
   hypotheses go into the directions ledger and round record.
10. **Promote.** A change becomes the new baseline only after it holds on the
    held-out task.

## Operational cadence

- Use `tools/exp_protocol_completion_monitor.py` as a slow batch wake-up, with
  a nominal one-hour poll (`--poll-seconds 3600`) and a threshold of eight
  terminal jobs. The cluster-side operator may reconcile more often; do not
  turn the planner into a minute-scale Slurm poller.
- Terminal state only wakes the loop. Harvest every attempt, then count only
  immutable-receipt, PTB-validator-complete, judge-clean cells toward the
  eight-cell analysis window. If the window is still short, restart the hourly
  monitor on the remaining relevant jobs.
- If a partial harvest occurs while the detector is still live below its
  terminal threshold, keep that process and its cumulative watched IDs; do not
  reset the counter merely because those attempts were harvested. Track NEW
  clean cells separately. [Window04's five-cell checkpoint](../../doc/exp_protocol_iterations/2026-09-04-window04-accumulation.md)
  shows why removing five terminal IDs and waiting for eight more would delay
  the intended trigger from eight to thirteen total terminals.
- After repairing operator state parsing, audit the full in-scope receipt queue
  for historical terminals the old parser skipped, not only the triggering job.
  Inspect available bundles before classifying them: a cancelled/requeued job
  can retain incomplete spillover evidence. The [cancelled-job backfill](../../doc/exp_protocol_iterations/2026-09-03-cancelled-backfill.md)
  recovered30 terminal records but added zero clean cells.
- Monitoring cadence does not weaken queue discipline: keep at least eight
  independently specified, validated cells held as `PENDING(JobHeldUser)` and
  release already-safe downstream work asynchronously when its scientific and
  ownership gates are satisfied.
- A receipt's `state: held` records intent, not the current scheduler hold.
  Count the floor from live `JobHeldUser` reasons; investigate any external
  release that disagrees with the committed queue before a screen can start.
- Recompute the held floor after every planned release/withdrawal. Replacing
  an old four-cell block with a new four-cell receipt adds no net buffer; a
  large current backlog can still be insufficient for the following wave.
  Prepare justified independent work before that release, without inventing
  filler or treating a checked manifest as an already-held receipt. The
  [operator dependency view](../../doc/exp_protocol_iterations/operator-state.md)
  records the current exact-ID arithmetic and prerequisites.
- For local Claude dispatch and liveness checks, follow
  `doc/reference/exp_protocol_local_claude_analysis.md`: separate the prompt
  from variadic tool arguments and verify the same session in the same
  execution-permission context before restarting it.

## Rules

- **Never** weaken a rule to raise a score without writing down why the rule
  was wrong, not just inconvenient.
- **Never** iterate on the held-out task.
- **Never** compare cells that differ in anything but the protocol.
- **Never** install this skill into a scientist's directory; `install` refuses,
  do not work around it.
- A round with no change is a valid round. Record it.

## What a good round looks like

Baseline `a1b2c3d` vs candidate `e4f5g6h` (adds `stop_token_consistent`), 3 seeds
each on gsm8k / gemma-3-4b-pt / opus-5 high. Candidate: `pitfalls_cost_h`
down from 2.1 to 0.4 summed over cells, accuracy +0.02 mean (inside noise),
two cards in the baseline cells cite eos in `pitfalls_hit`, none in the
candidate's. Decision: promote after the held-out task (aime2025) shows the
same `pitfalls_cost_h` drop. Recorded as round-03.
