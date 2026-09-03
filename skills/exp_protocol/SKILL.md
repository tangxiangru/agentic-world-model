---
name: exp_protocol
description: Use before, during, and after every model-training or evaluation experiment. Write the card first, run the pre-flight checks, lock, then train; close the card after. Keeps a run from producing a clean-looking wrong answer, and leaves a record another agent can rerun and understand.
---

# Experiment protocol

## Why this exists

Under a fixed executor, six historical recipes reproduced about a tenth of the
score spread the corpus attributes to them; roughly 90 % of a run's outcome is
how it was executed, not what was tried. The three defects a CPU dry run caught
in one such rerun — targets ending in the wrong stop token, rows longer than
`max_seq_len`, a chat template the sandbox could not reach — each produced a
clean-looking wrong answer and exited 0. This protocol is the checklist that
catches that class of failure before GPU time is spent, and the record that
lets the next experiment start from where this one ended.

It tells you how to do an experiment correctly. It does not tell you what to
run; that is your call.

## The unit: one card, one experiment

One card = one problem, one hypothesis, one intervention, one result. A sweep
is several cards. A three-stage pipeline is three cards, each naming the
previous card's output as `setup.parent_checkpoint`. Cards live in
`{dir}/memory/cards/exp-NN.yaml`; `{dir}/memory/index.md` is one line per card.

Seven sections. **0–4 are written before the launch command runs; 5–6 after.**

| # | section | answers |
|---|---|---|
| 0 | `situation` | what state you were in when you decided this: hours used/left, the incumbent, what observation triggered this, what alternatives you rejected and why, what pitfalls you already hit |
| 1 | `problem` | what is going wrong, with evidence paths |
| 2 | `hypothesis` | what you expect to change, against what, and what would prove you wrong |
| 3 | `setup` | parent checkpoint, data, method, the exact command, output dir, which checkpoints you will keep |
| 4 | `evaluation` | the protocol (`n`, dev set, seed) and the comparator scored under it |
| 5 | `result` | what physically happened |
| 6 | `conclusion` | verdict, decision, next step |

`card.template.yaml` in this directory is the annotated template;
`example-card.yaml` is one filled in.

## Workflow

```bash
awm exp_protocol index --dir {dir}          # 1. read memory/index.md and the starting points first
awm exp_protocol new --dir {dir}            # 2. writes exp-NN.yaml with every required field empty, prints the questions
#    edit sections 0-4
awm exp_protocol check --dir {dir} exp-NN   # 3. repeat until no ERROR lines and no questions; "ok (N warnings, advisory)" is ok
awm exp_protocol lock  --dir {dir} exp-NN   # 4. runs preflight; refuses on any FAIL; pins sections 0-4, the script, and the data;
#                                                 if a world-model agent is attached it asks for the verdict and WAITS for it (see below)
#    launch your command exactly as written in setup.command.argv — only after lock has returned
#    keep checkpoints as setup.checkpoints says
#    evaluate the output under evaluation.protocol; fill sections 5-6
awm exp_protocol close --dir {dir} exp-NN   # 5. validates 5-6, re-checks the lock, rebuilds the index
```

`awm exp_protocol preflight --dir {dir} exp-NN` can be run on its own any time.
Read its reminders: they are the pitfalls no check can catch for you.

Two escape hatches, both recorded in the lock file where the meta loop counts
them: `lock --relock "<reason>"` when you must change sections 0–4 after a
lock (the previous hash is kept), and `lock --override <check>="<reason>"`
when a pre-flight check is wrong for your data (say why). Neither is free:
a re-lock or an override without a real reason is what the record will show.

## If a world-model agent is installed

You interact with the WMA only through the thin `awm wma` client. The WMA's
own skill, priors, historical corpus and execution process are private to its
sidecar and are not scientist inputs: do not look for or try to read them.

**The verdict is part of the lock.** When a world-model agent is attached,
`awm exp_protocol lock` asks it about the card the moment the lock is written
and then waits — up to 20 minutes, printing a heartbeat every 30 s — until the
verdict lands beside the card (`exp-NN.verdict.json`). The last lines it prints
are `verdict: L0=… L1=… L2=… L3=…; first precondition: …` and the verdict path.
It is advice; you decide. Four things matter about how you use it:

- **Do not start the run before `lock` has returned.** Run `lock` with a
  long tool timeout (25 min) or in the background and use the wait to prepare
  everything else — build data, dry-run the evaluation command on the parent,
  write the launch line — but the training or evaluation command in
  `setup.command.argv` starts only after the verdict line (or the timeout /
  failure line) has been printed. A run launched while the verdict was still
  pending is a protocol violation and the record will show it.
- **Read the whole verdict before launching.** If it says *verify first*, that
  is usually minutes on CPU that can save an hour on the GPU: do it, or write in
  `situation.alternatives_rejected` of this card why you did not, before `close`.
  If it says *no* or *defer* and you disagree, run anyway and note why in the
  same place; that disagreement is the record the estimator learns from. If the
  verdict changes the plan, edit sections 0–4 and `lock --relock "<reason>"`:
  the estimator is asked again about the changed card.
- **A failed or timed-out review is not a verdict.** `lock` says so and records
  it; you may launch. Do not retry the review by hand before launching.
- **If it answers "no world-model agent is attached to this cell", that is the
  whole answer.** No verdict will come; do not retry, do not look for one. Carry
  on with step 5.

`awm wma review --dir {dir} exp-NN` asks again about an already-locked card and
waits the same way; `--background` queues one or more locked cards and returns
at once — for a second opinion on several candidate cards, never as a substitute
for the wait inside `lock`. `awm wma status --dir {dir}` shows what is in. The
only WMA files you should see are request/status records and
`exp-NN.verdict.json` beside the card. Missing `skills/wma` is intentional.

## The rules

1. **Sections 0–4 before the command runs.** A hypothesis written after the
   result is a description of the result. `lock` pins them; `close` re-checks.
   A material change after lock is a new card, not an edit.
2. **A comparator is measured under the same protocol.** Same `n`, same dev
   set, same seed, and the path of that eval goes in `evaluation.comparator.path`.
   A number from a different `--limit` is not a comparator.
3. **Unknown is `null`, never a guess.** `check` asks for what is missing;
   answer it or leave it null. Do not invent an evidence path.
4. **A target is not a hypothesis.** "Reach 85 %" is rejected as a claim.
   Write the intervention, the metric, the direction, and `falsified_if`.
5. **Failed and killed runs get closed too.** `result.execution: failed` with
   the traceback tail is a result; `conclusion.verdict: inconclusive` is a
   conclusion. A card is never deleted.
6. **Keep the checkpoints you said you would.** `setup.checkpoints.keep` is a
   promise to the next card, which may start from them. Record what you kept
   in `result.checkpoints_kept`.
7. **The benchmark's test copy is input to the contamination checker only.**
   No item from it in `failure_examples`, `watch_set`, or training data.
8. **Write the situation honestly.** If you are doing this because 40 minutes
   are left, say so in `situation.trigger`; the next reader must not mistake a
   last-ditch gamble for a validated method. Every smoke run, OOM, and wrong
   path you hit before this launch goes in `situation.pitfalls_hit` with the
   hours it cost.

## Pitfalls

`pitfalls.yaml` in this directory lists the known ways a run produces a
clean-looking wrong answer. Each names the preflight check that catches it,
or says `check: null` when only you can. For any card that trains on target
text (`family` sft / rft / dpo / grpo / distill), `setup.method.stop_token` and
`hyperparams.max_seq_len` are **required** — `check` refuses the card without
them, because the eos and truncation checks are the two that each cost a whole
run. `setup.method.answer_marker` is advisory: declare it when the grader reads
one (gsm8k, aime), leave it null when it does not. The data checks read
the first 500 rows of each jsonl file and look for the target under
`completion`, `target`, `output`, `response`, then `messages[-1].content`,
then `text` / `answer`; if your field is named differently the check SKIPs
and says so — rename the field or check by hand.

When you lose time to something not in the list, record it in the next card's
`situation.pitfalls_hit`. That is how the list grows.

## Budget

The protocol should cost you under five minutes per card. If it is costing
more, the card is doing too much: split it.

## Optional: Claude Code Stop hook

`hooks/stop_open_cards.py` blocks the end of a turn once if a locked card has
no conclusion, so a finished run is not left unclosed. Standard library only.
Install by adding it to your task dir's `.claude/settings.json` under `Stop`;
it is not installed by default and does nothing for Codex.
