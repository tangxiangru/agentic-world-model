---
name: consult
description: Use for every message from the research scientist. Turns a plan, or a plan plus new results, into the fixed consult response — card, verdict, eval plan, suggestion, reasons — grounded in the past experiments you are allowed to read, then logs it.
---

# Consult

One procedure for every message. Do not skip steps; do not add ceremony.

## 1. Read the request and the config

- `cat $AWM_SESSION_DIR/wm/config.json` — `arm`, `prior_runs_root`, `memory_root`, `session_dir`.
- Decide the `stage`: first mention of an experiment → `plan`; results reported → `running`;
  "shipping / final model / done" → `shipped`.
- Save the scientist's message verbatim to `$AWM_SESSION_DIR/wm/tmp/request.txt` (you need it for `awm wm log`).

## 2. Draft the card

```bash
awm wm draft-card --text "$(cat $AWM_SESSION_DIR/wm/tmp/request.txt)" [--card-id exp-NN]
```

Gives a skeleton from the launch command, data files and workspace, and the list
of gaps. Open the scientist's training script and data prep if the command names
them; argparse defaults count as established (say so). Reuse the `card_id` of
an experiment you have already seen (`ls $AWM_SESSION_DIR/wm/cards`); a
materially different plan is a new card. On a re-consult, add the reported
numbers to `card.results`, and verify any eval output path you were given:

```bash
awm wm read-eval /home/ben/task/eval_step300.json
```

## 3. Search the past

- `retrieval` / `llm`: `awm wm search --text "<plan>" --k 5` — nearest cards with best-vs-parent deltas and decisions.
- `traj` / `llm`: open `prior_runs_root/INDEX.md`; pick runs on the **same base model** with a similar recipe (data, method, steps); read their complete `solve_out.txt` trajectories for what they did and observed, `metrics.json` for the final score, and `time_taken.txt` for duration. Files mentioned inside a trace are not present unless they are one of those three. Three to five runs read carefully beat twenty skimmed.
- `null`: skip this step; your verdict is `CANNOT_DECIDE` and your plan is the default.

Write down, per run you rely on: path, what it did, its trajectory of numbers,
its final score. These become `based_on` and `reasons`.

## 4. Decide

- **Verdict.** Do the comparable runs agree? If the runs like this one mostly ended clearly above the parent → `SURE_WILL_WORK`; mostly at or below → `SURE_WONT_WORK`; mixed, few, or none → `CANNOT_DECIDE`. Confidence is the fraction of your evidence pointing one way, tempered by how comparable it is; `SURE_*` needs ≥ 0.75 and citations. Prediction: the comparable runs' deltas → mean and spread; `null` when you have fewer than two.
- **Eval plan.** Where did the comparable runs' intermediate numbers separate the good from the bad? Put the points there, with the number the scientist should be at. Otherwise `awm wm eval-plan --steps N --parent <value>`.
- **Suggestion.** `KEEP_RUNNING` (say which result would settle it and when), `TERMINATE` (say why continuing will not pay, and what to do instead), or `ADJUST` (one change + the run that shows it matters).

## 5. Answer and log

Write the JSON to `$AWM_SESSION_DIR/wm/tmp/response.json`, then:

```bash
awm wm log --response $AWM_SESSION_DIR/wm/tmp/response.json --request $AWM_SESSION_DIR/wm/tmp/request.txt
```

If it rejects the response, fix it — do not send an unlogged reply. Then
`SendMessage` to the scientist's address: the ```json block, then a summary of
at most five lines — verdict and confidence, predicted delta, the next
evaluation point and the number to beat, the suggestion, the top reason.

If `stage` is `shipped`: `awm wm outcome --card exp-NN --final <value> --shipped <checkpoint>` before replying.

## 6. Wait

`sleep 120` via Bash, then finish the tool round so queued messages drain.
Repeat until the run is over.
