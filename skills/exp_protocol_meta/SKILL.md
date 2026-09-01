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
itself are a separate, human decision.

## The loop

1. **Pick the variants.** A variant is a commit of `skills/exp_protocol/` (plus
   the preflight code it relies on). Name each by its short sha. Two or three per
   round; one of them is always the current baseline.
2. **Fix everything else.** Same task, same base model, same scientist model and
   effort, same `PTB_NUM_HOURS`, across variants. The protocol is the only
   thing that differs between cells.
3. **Seeds.** At least two cells per variant. Run-to-run variance on
   PostTrainBench is large; one cell per variant decides nothing.
4. **Held-out task.** One task is never used for iteration; it is run only to
   confirm a change generalises before it becomes the baseline.
5. **Launch.** Every scientist cell must run
   `awm exp_protocol install --target /home/ben/task --tool <claude|codex>`
   before the prompt is handed to the agent, with `AWM_EXP_PROTOCOL_DIR`
   pointing at the variant's checkout of `skills/exp_protocol`. Use the
   committed batch launcher (`awm ptb`, manifests under
   `experiments/posttrainbench/`); do not hand-craft sbatch files.
6. **Collect.** When the cells finish:
   ```bash
   awm exp_protocol collect <result>/<cell>/task ... --csv > round-NN.csv
   ```
   `metrics.md` defines each column and what a move in it means.
7. **Analyse.** Per variant: mean and range of `accuracy`; sum of
   `pitfalls_cost_h`; `n_locked_open` (cards started and abandoned);
   `fields_filled`. Read three cards per variant by hand — the numbers say
   whether, the cards say why.
8. **Decide one change.** The change must be traceable to something you saw:
   a pitfall that cost hours and has no check; a rule scientists skipped; a
   field that stayed empty. Write it up first, then make it.
9. **Record.** Copy `iteration_record.template.md` to
   `doc/exp_protocol_iterations/<date>-round-NN.md`, fill every section, commit
   it with the change in the same commit.
10. **Promote.** A change becomes the new baseline only after it holds on the
    held-out task.

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
