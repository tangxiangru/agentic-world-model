---
name: exp_protocol
description: Use before, during, and after every model-training or evaluation experiment. Compare real training proposals before choosing, then write the card, preflight, lock, record the action, launch with run, and close. Keeps execution and decisions reproducible.
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

## Configured decision mode

Read `awm_sandbox.json` before the workflow. Its frozen `decision_mode` is
`single`, `multi-self`, or `multi-joint` (legacy setups without this field use
`multi-joint`). Do not change this study configuration.

- `single`: write the selected formal card directly; omit the candidate-set
  step below. All preflight, blocking lock, action and `run` checks still apply.
- `multi-self`: prepare the same real briefs below and choose yourself. The
  `compare` command freezes their version and returns `not_requested`; it makes
  no joint model call. The selected card still receives its normal WMA review
  if a sidecar is attached.
- `multi-joint`: use the joint comparison below before choosing, then obtain
  the selected card's separate blocking review.

## Before choosing the next training (multi modes)

In either multi mode, before committing the next major training budget, prepare usually 2–3 real
candidate briefs in a decision file. This step also applies when no WMA is
attached, so a comparison measures the value of its advice on the same
candidate set. Do not invent extra candidates: one is allowed with a
`singleton_reason`. Routine packaging, evaluation, and implementation repairs
do not need a new candidate set.

```bash
awm wma propose --dir {dir}                 # creates memory/decisions/decision-NN.proposal.json
# edit the briefs; record scientist_preference BEFORE asking for comparison
awm wma compare --dir {dir} decision-NN     # freeze briefs; joint call only in multi-joint mode
# read the comparison, or its recorded unavailable/failed/timed-out status; choose yourself
awm wma choose --dir {dir} decision-NN --candidate A --reason "why A is the best use of this budget"
```

Use `proposal.example.json` in this directory for the file shape; its paths,
numbers, and choices are synthetic. Share the incumbent, hours left, and
evidence once. For each candidate state the parent, data/objective/schedule
change, hypothesis, train/eval hours, cost basis, evidence, uncertainty, and
what observation would change the next decision. Mark unprepared inputs
honestly; these are briefs, not a requirement to build several datasets or
run several trainings before choosing.

Keep the pre-comparison preference unchanged. Record the final choice and
reason even when you agree with it or no comparison was available. Once the
selected formal card exists, bind it with `choose ... --card exp-NN`.
This binds the plan and its declared script/data/config bytes. After a repair
changes them, confirm the choice again with `choose --card` and a reason; if
the candidate brief itself changes, compare the new brief first. Use
`choose ... --decline --reason "..."` if none of the real candidates should run.
Comparison is advice, not launch approval: the selected card still follows
the complete workflow below. Independent single-card second opinions are not
a joint comparison of alternatives.

## Workflow

```bash
awm exp_protocol index --dir {dir}          # 1. read memory/index.md and the starting points first
awm exp_protocol new --dir {dir}            # 2. writes exp-NN.yaml with every required field empty, prints the questions
#    edit sections 0-4
awm exp_protocol check --dir {dir} exp-NN   # 3. repeat until no ERROR lines and no questions; "ok (N warnings, advisory)" is ok
awm exp_protocol lock  --dir {dir} exp-NN   # 4. runs preflight; refuses on any FAIL; pins sections 0-4, the script, and the data;
#                                                 if a world-model agent is attached it asks for the verdict and WAITS for it (see below)
awm wma act --dir {dir} exp-NN --action proceed --reason "review considered; why this version should run"
awm exp_protocol run --dir {dir} exp-NN    # 5. launches the locked argv, only after lock has returned;
#                                                 checks current hashes and a current proceed record, records launch and exit
#    keep checkpoints as setup.checkpoints says
#    evaluate the output under evaluation.protocol; fill sections 5-6
awm exp_protocol close --dir {dir} exp-NN   # 6. validates 5-6, re-checks the lock, rebuilds the index
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
- **Read the whole verdict before launching.** Record how it changes your
  action with `awm wma act` before executing that action, including when you
  decline advice inside a `yes` verdict. If it says *no* or *defer* and you
  disagree, record the reason, then proceed through `run`. If the verdict
  changes the plan, edit sections 0–4 and `lock --relock "<reason>"`: the
  estimator is asked again about the changed card. Record a new `proceed`
  after the new lock returns; an old action does not authorize a new version.
- **A failed or timed-out review is not a verdict.** `lock` says so and records
  it; you may record `proceed` with that status in the reason and launch with
  `run`. Do not retry the review by hand before launching.
- **If it answers "no world-model agent is attached to this cell", that is the
  whole answer.** No verdict will come; do not retry, do not look for one. Carry
  on with the action record and step 5.

Record a `probe`, `repair`, `decline`, or `abandon` action when that is your
decision; none substitutes for the final `proceed` needed by `run`. For example:

```bash
awm wma act --dir {dir} exp-NN --action probe --suggestion precondition-1 --reason "compare existing checkpoints to choose the parent"
# perform the check; if the original parent and locked plan still stand, record why to proceed
awm wma act --dir {dir} exp-NN --action proceed --reason "comparison retained the original parent; old-data plateau does not invalidate the proposed new mixture within the remaining budget" --evidence memory/checkpoint-comparison.json
```

`--evidence PATH` may be repeated. Action records are immutable; add a new
event when the observation or decision changes. Record cancellations too and
close unrun cards with their actual status. An unrun recipe's endpoint remains
unknown. A short training or an old checkpoint plateau may change confidence
or budget priorities, but does not by itself prove a changed full recipe has
failed. A real implementation blocker can still require repair before launch.

`awm wma review --dir {dir} exp-NN` asks again about an already-locked card and
waits the same way; `--background` queues one or more locked cards and returns
at once — for a second opinion on several candidate cards, never as a substitute
for the wait inside `lock`. `awm wma status --dir {dir}` shows what is in. The
only WMA files you should see are request/status records and
`exp-NN.verdict.json` beside the card, plus your decision and action records
and the joint comparison under `.wma/comparisons`. Proposals and choices are
under `memory/decisions`. Missing `skills/wma` is intentional.

## The rules

1. **Sections 0–4 before the command runs.** A hypothesis written after the
   result is a description of the result. `lock` pins them; `close` re-checks.
   Launch `setup.command.argv` through `awm exp_protocol run`; do not bypass
   its hash and action checks with a manual launch. A new hypothesis is a new
   card; implementation repairs to the existing plan require a re-lock.
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

Keep card bookkeeping short. Record comparison/review time and probe costs as
part of the session budget. Estimate training and evaluation costs from this
trajectory's measured throughput when available, and update stale estimates.
Before ending early, record the real options that fit the remaining time and
why keeping the incumbent is preferable. More completed SFTs is not a goal by
itself; there is no minimum training-count target.

## Optional: Claude Code Stop hook

`hooks/stop_open_cards.py` blocks the end of a turn once if a locked card has
no conclusion, so a finished run is not left unclosed. Standard library only.
Install by adding it to your task dir's `.claude/settings.json` under `Stop`;
it is not installed by default and does nothing for Codex.
