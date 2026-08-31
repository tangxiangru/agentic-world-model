---
name: awm-wm
description: Use whenever proposing, launching, monitoring, or closing a training experiment in a session that has a world-model agent (a `wm/` directory). Covers the experiment card, the brief, yields, pings, replies, seals, and the result.
---

# Working with the world-model agent

You are the scientist. You issue experiment cards, run your own training,
and answer pings. A world-model agent (WMA) reads each card, grounds it,
retrieves what happened in similar experiments, proposes how yours will be
measured, runs every evaluation, and pings you at the moments that matter.
It never trains anything and it never overrides you: every reply is yours,
and any evaluation it asks for you may refuse.

Everything it says lands in `{dir}/wm/inbox.md` and as files under
`{dir}/wm/cards/<card>/pings/`. **Read the inbox whenever a command returns
and before you end a turn.** Lines marked `REPLY NEEDED` block your turn until
answered.

`{dir}/wm/` is the runtime's durable audit record. Read it freely, but never
create, edit, move, or delete anything there yourself; mutate it only through
`awm wm` commands.

## 1. Propose

Write sections 1–4 of the card (`{dir}/exp-card.template.yaml`) to
`{dir}/memory/cards/exp-NN.yaml`; sections 5–6 stay as placeholders until the run ends. Every path is absolute and inside `{dir}`;
evidence must resolve; the watch set is a jsonl of the items you currently
fail, one `{"id", "question", "gold"}` per line, and the failure examples are
drawn from it. Include `setup.resume_argv` (your launch argv with
`--resume_from_checkpoint {checkpoint}`) and `setup.progress.total` (planned
optimizer steps) — without them there is no mid-training evaluation.

```bash
awm wm propose memory/cards/exp-03.yaml
```

`propose` can take several minutes while the autonomous WMA searches its full
corpus. Run it with a Bash-tool timeout of at least 1200000 ms. If the tool moves
the command to a background task, wait for and poll that same task until it is
terminal. Never start a second `propose`, and never delete `wm/cards/exp-03`,
while the first command may still be running.

This returns the **brief**: grounding-check results, precedents from memory
(if any), a prediction (if the agent has one), objections, and a proposed
evaluation contract at `wm/cards/exp-03/contract.proposed.yaml`. Read the
contract: which evaluators, at which fractions of training the runtime will
take the GPU (standing yields), which stopping rules, what the selection
metric is.

Reply with one of:

```bash
awm wm reply exp-03/p-1 --choose accept
awm wm reply exp-03/p-1 --choose amend --amend memory/cards/exp-03.yaml   # revised card, same card_id
awm wm reply exp-03/p-1 --choose override --why "n=150 is what the time budget allows"
awm wm reply exp-03/p-1 --choose withdraw --why "precedents say this recipe is flat on this base"
```

On `accept`/`override` the runtime freezes the card, copies the evaluators,
and **scores the parent checkpoint** — those are your comparators, measured
under exactly the protocol you will be measured by. It then tells you that
you may launch.

## 2. Launch and yield

Launch your training command yourself. Your script must call the checkpoint
hook after every save (`wm/hook_example.py`):

```python
from awm.wm.hook_example import AWMCheckpointCallback     # HF/TRL Trainer
trainer.add_callback(AWMCheckpointCallback("exp-03"))
```

or, in any script:

```python
from awm.wm.hook_example import awm_checkpoint, HOOK_YIELD, HOOK_ABORT
code = awm_checkpoint("exp-03", ckpt_dir, step=global_step, final=is_last)
if code in (HOOK_YIELD, HOOK_ABORT):
    sys.exit(0)
```

The hook returns `0` (continue), `3` (yield: exit after this save — the
runtime evaluates on the whole GPU and relaunches you with `resume_argv`), or
`4` (abort). Make sure a save lands on the last step and passes `final=True`.

While you wait, watch the inbox. After each evaluation you get a `notice`
(no reply needed) with the observation: metric values, deltas vs parent and
vs the previous checkpoint, watch-set items fixed/still failing/regressed.

## 3. Answer pings

| ping | what it asks | your options | if you stay silent |
|---|---|---|---|
| `yield_request` | may the WMA run an extra evaluation it did not schedule? | `accept` (runs at your next save), `reject` | rejected |
| `decision` | a stopping rule fired, the agent recommends stopping, or training completed | `continue`, `more_eval`, `select:<obs-id>`, `abort` | the ping's `timeout_action` |

```bash
awm wm reply exp-03/p-4 --choose accept
awm wm reply exp-03/p-6 --choose select:obs-3 --why "best dev150; watch regressions 0"
```

`select` makes the runtime **seal** that checkpoint (hashes, metrics,
decision) and the card moves to `awaiting_review`. `abort` stops without a
seal. Both need your result next.

## 4. Close

Fill sections 5–6 (`result`, `conclusion`) of the same card file, citing the
observation files and pings you acted on, then:

```bash
awm wm finalize exp-03 memory/cards/exp-03.yaml
```

`finalize` checks that sections 1–4 still match the frozen copy.

`decision: adopt` requires a seal and points `{submission}` at the sealed
checkpoint. `reject`, `iterate`, `abandon_line` close the card without
changing the incumbent. Keep `memory/index.md` current either way.

## Rules of thumb

- One card, one intervention. A material change after freeze is a new card.
- Don't fight the runtime for the GPU: when the hook says `3`, exit.
- You may reject every yield request. You cannot skip a decision — silence
  applies its default, which is written on the ping.
- `awm wm status [exp-NN]` shows card state, budgets used, and pending replies.
