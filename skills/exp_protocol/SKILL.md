---
name: exp_protocol
description: Use before, during, and after model-training or evaluation experiments to check actual execution and evidence, avoid known process failures, and preserve reproducible cards and checkpoint lineage. Write, check and lock before running; close afterward. Does not select research recipes.
---

# Experiment protocol

## Why this exists

A successful command and a complete card can still describe the wrong data,
decode settings or comparison. This protocol combines execution discipline,
process knowledge from observed failures, and the evidence another scientist
needs to reproduce or continue an experiment.

It tells you how to execute and substantiate an experiment. You choose the
research direction, recipe, useful diagnostics and budget tradeoffs. A past
successful recipe is a precedent, not a required method. An automated
checklist must not disguise an uncertain recipe recommendation as a fact.

Read the relevant sections of [Process checks](process_checks.md) when preparing
data, training, sampling, comparisons or exports. They describe checks and
evidence to produce; only the named implemented preflight checks are automatic.
Use existing artifacts and cheap CPU checks first. Extra model probes still
need a prospective card and a reason to spend their budget.

## The unit: one card, one experiment

One card = one problem, one hypothesis, one intervention, one result. A sweep
is several cards. A three-stage pipeline is three cards, each naming the
previous card's output as `setup.parent_checkpoint`. Cards live in
`{dir}/memory/cards/exp-NN.yaml`; `{dir}/memory/index.md` is one line per card.

Training/evaluation smokes, benchmarks, retries and graded generation are
covered too: declare their commands before running and check/lock the covering
card. A retrospective `smoke_runs` entry does not supply a missing earlier
lock. Pure file inspection and tokenization do not train or evaluate a model.

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
awm exp_protocol lock  --dir {dir} exp-NN   # 4. runs preflight; refuses on any FAIL; pins sections 0-4, the script, and the data
#    launch your command exactly as written in setup.command.argv
#    keep checkpoints as setup.checkpoints says
#    evaluate the output under evaluation.protocol; fill sections 5-6
awm exp_protocol close --dir {dir} exp-NN   # 5. validates 5-6, re-checks the lock, rebuilds the index
```

`awm exp_protocol preflight --dir {dir} exp-NN` can be run on its own any time.
Read its reminders: `check: null` means guidance, not an implemented check.
PASS describes the check's actual scope; it does not certify every command,
effective setting, measured result or final artifact.

Two escape hatches, both recorded in the lock file where the meta loop counts
them: `lock --relock "<reason>"` when you must change sections 0–4 after a
lock (the previous hash is kept), and `lock --override <check>="<reason>"`
when a pre-flight check is wrong for your data (say why). Neither is free:
a re-lock or an override without a real reason is what the record will show.

## If a world-model agent is installed

Only use this optional integration if the installed CLI actually provides
`awm wma review`. Some experiment sandboxes ship no WMA; do not install it,
search for it or make it a launch dependency to follow this skill.

`awm wma review --dir {dir} exp-NN --background` asks an estimator what your locked card will do: one line —
*worth running now*, why, what to verify first, a cheaper variant if there is
one. It is advice; you decide. Three things matter about how you use it:

- **Always `--background`.** It returns at once and the verdict lands beside the
  card (`exp-NN.verdict.json`) when it is done. Never sit waiting for it; keep
  preparing the launch. `awm wma status --dir {dir}` shows what is in.
- **Batch your candidates.** If you have several cards you could run next, review
  them in one call — `awm wma review --dir {dir} exp-05 exp-06 exp-07 --background`
  — they run in parallel and you get a ranking, not three separate waits.
- **Do not wait for it to launch.** If the verdict has not arrived by the time you
  would start the run, start the run. Time is the budget; the verdict is worth
  exactly what it saves you, and nothing when it costs you the launch.

Read the verdict when it arrives. If it says *verify first*, that is usually
minutes on CPU that can save an hour on the GPU. If it says *no* and you disagree,
run anyway and note why in `situation.alternatives_rejected` of the next card;
that disagreement is the record the estimator learns from.

## The rules

1. **Sections 0–4 before the command runs.** A hypothesis written after the
   result is a description of the result. `lock` pins them; `close` re-checks.
   A changed hypothesis/intervention needs a new card. A repair to its command,
   script or data requires a reasoned relock **before** the repaired run.
2. **Bind comparisons to actual measurements.** Record actual scored IDs/n,
   input and checkpoint identity, seed policy and effective evaluation settings.
   Match them for a controlled comparison; an intentional decode or other
   treatment contrast must declare what differs. A path, matching scalar or
   requested `--limit` does not prove these facts. If a measurement card will
   produce both sides, leave unavailable comparator value/path null, declare
   the two planned reads before lock, and record actual evidence in `result`
   afterward. Do not evaluate first merely to make preflight pass. See
   [current v2 limits](process_checks.md#current-v2-evidence-limits).
3. **Unknown is `null`, never a guess.** `check` asks for what is missing;
   answer it or leave it null. Do not invent an evidence path.
4. **A target is not a hypothesis.** "Reach 85 %" is rejected as a claim.
   Write the intervention, the metric, the direction, and `falsified_if`.
5. **Failed and killed runs get closed too.** `result.execution: failed` with
   the traceback tail is a result; `conclusion.verdict: inconclusive` is a
   conclusion. A card is never deleted.
6. **Keep the checkpoints you said you would.** `setup.checkpoints.keep` is a
   promise to the next card, which may start from them. Record what you kept
   in `result.checkpoints_kept`, with identity/parent information in a linked
   artifact record. Rejecting a hypothesis does not silently cancel that
   promise. You decide what is worth retaining or combining within the budget;
   the protocol provides lineage and comparison evidence, not that choice.
7. **The benchmark's test copy is input to the contamination checker only.**
   No item from it in `failure_examples`, `watch_set`, or training data.
8. **Write the situation honestly.** If you are doing this because 40 minutes
   are left, say so in `situation.trigger`; the next reader must not mistake a
   last-ditch gamble for a validated method. Every smoke run, OOM, and wrong
   path you hit before this launch goes in `situation.pitfalls_hit` with the
   hours it cost.
9. **Your turn is the session.** In the single-session PTB scaffold, when
   your turn ends the session ends, and every background process you started
   dies with it — a training run included. There is no next turn. Never end
   the turn while a run is alive. Track the actual producer, retain its exit
   status and verify this invocation's outputs before evaluation/closure.
   Use bounded observation calls and a launch lifetime that survives their
   timeout; a long foreground command can be killed by the tool timeout.
   A quiet log, empty GPU or existing `config.json` alone is not completion.
   If exit or artifact evidence is missing, record it as unverified rather
   than guessing success or launching a duplicate. See
   [run lifetime](process_checks.md#run-lifetime-and-recoverable-output).

## Evidence that remains useful after a negative result

Put actual outcome values and artifact paths in `result.measurements`,
`diagnostic_result` and `training_summary.notes`; use `conclusion` for what
they support and the next decision. Existing v2 fields can link a compact
JSON/Markdown evidence record; do not invent new required schema fields.

Report the declared task metric alongside relevant measured cost, failure,
format/termination or paired-error evidence. Distinguish a failed execution,
a contradicted claim, an inconclusive comparison and a usable artifact. Equal
accuracy does not establish equal errors; different errors do not guarantee
a useful combination. A gold-answer oracle union is an analytical upper
bound, not an achieved model score. Deciding which branch to pursue is yours.

Do not turn a newly noticed metric into retrospective proof of the original
claim. Record the opportunity and uncertainty for a possible future card.
Use permitted train/dev/probe sources; this does not relax rule7 or authorize
turning benchmark test examples, answers or IDs into training data.

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
and says so. Those checks do not execute your renderer/tokenizer; verify the
actual supervised tokens separately. If a correct renderer appends the stop
token, document that evidence and use a reasoned override of a raw-field false
alarm rather than appending a duplicate token to satisfy it. Sampled raw checks
and character-based length estimates are not whole-corpus runtime proofs.

When you lose time to something not in the list, record it in the next card's
`situation.pitfalls_hit`. That is how the list grows.

## Budget

Keep bookkeeping short by linking real artifacts and reusing recorded facts.
Investigate repeated friction instead of multiplying cards to meet a time
target. Separate cards when the hypothesis/intervention changes, not merely
because recording took five minutes. Negative results and unspent budget need
honest explanations; neither filling the budget nor maximizing card count is
a goal.

## The Stop hook

`hooks/stop_open_cards.py` blocks the end of a turn while a locked card has no
conclusion, up to twelve times, and each time says why (rule 9) and how to wait.
It is installed into `.claude/settings.json` by `awm sandbox setup --exp-protocol
--stop-hook`. Closing the card releases it; if the CLI is unavailable, filling
`result` and `conclusion.decision` in the YAML by hand counts as closed.
Use the template's block-style top-level `conclusion` mapping. The hook checks
that decision field, not live processes, exit status, result correctness or
artifact loadability. An absent block is not proof of those properties, and
hand-filled closure is not CLI-validated closure. Standard library only;
does nothing for Codex.
