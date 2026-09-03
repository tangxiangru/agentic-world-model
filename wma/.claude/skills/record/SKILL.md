---
name: record
description: Use for every message from the research scientist. Turns a launch plan, reported results, or a ship notice into the fixed record response — card, questions, ack — snapshotting what the launch names, then logs it. Never gives advice.
---

# Record

One procedure for every message. Do not skip steps; do not add ceremony; do
not advise.

## 1. Read the request and the config

- `cat $AWM_SESSION_DIR/wm/config.json` — `mode` (must be `record`), `session_dir`.
- Decide the `stage`: first mention of a launch → `plan`; results reported → `running`;
  finished, failed, abandoned, or "shipping / final model / done" → `closed`.
- Save the scientist's message verbatim to `$AWM_SESSION_DIR/wm/tmp/request.txt`
  (you need it for `awm wm record`).
- If the message asks for an opinion, a prediction, or a recommendation, your
  reply will say, in one line, that you only keep the record — and you still
  record whatever plan or results the message contains.

## 2. Draft the card

```bash
awm wm draft-card --text "$(cat $AWM_SESSION_DIR/wm/tmp/request.txt)" [--card-id exp-NN]
```

Gives a skeleton from the launch command, data files and workspace. Reuse the
`card_id` of a launch you have already recorded (`ls $AWM_SESSION_DIR/wm/cards`);
a materially different plan — changed script, changed hyperparameters, changed
data — is a **new card**, with `setup.parent_checkpoint.origin` naming the card
that produced its starting checkpoint (`base_model` if it starts over).

Then fill the card from the workspace, not from imagination: open the training
script and the data prep the command names; argparse defaults count as
established (note in the card that they are defaults). Keep the scientist's own
words for `problem.statement` and `hypothesis.claim`; null if it gave none.

## 3. Snapshot (plan stage)

```bash
awm wm snapshot --card exp-NN /abs/path/train.py [/abs/path/prep_data.py ...]
```

At `plan` stage, snapshot every file the launch command names plus anything the
card's reproducibility depends on (data-prep scripts, config files). The
scientist edits files in place; the snapshot is what keeps the card true.

## 4. Verify results (running / closed stage)

Reported numbers must be tied to files:

```bash
awm wm read-eval /home/ben/task/eval_step300.json
```

Record the file's numbers with `value`, `n`, and `source`. If the scientist's
message and the file disagree, record the file and note the discrepancy in the
card. A crash or a killed run is recorded the same way: `result.execution`,
the log path, the error line.

## 5. Answer and log

Write the JSON to `$AWM_SESSION_DIR/wm/tmp/response.json`, then:

```bash
awm wm record --response $AWM_SESSION_DIR/wm/tmp/response.json --request $AWM_SESSION_DIR/wm/tmp/request.txt
```

If it rejects the response, fix it — do not send an unlogged reply. It returns
`missing`: the sufficiency gaps still open. Pick the at-most-three most
important ones the workspace cannot answer and make them your `questions`.

Then `SendMessage` to the scientist's address: the ```json block, then the ack
line and the questions — nothing else. No commentary on the plan's merits.

If the scientist shipped: `awm wm outcome --card exp-NN --final <value> --shipped <checkpoint>`
before replying.

## 6. Wait

`sleep 120` via Bash, then finish the tool round so queued messages drain.
Repeat until the run is over.
