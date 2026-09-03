# Trace review: read every cell before deciding the round

`collect` says whether a variant moved a number; only the traces say why. In
Round 00 the protocol was executed to the letter — every card locked and
closed, every launch after its lock — and the arm still scored 0.07 below the
control. Nothing in the metrics explained it. The traces did: three of nine
protocol cells shipped Gemma's stock sampling `generation_config`, five of
seven rejection-sampling cards died in the same three vLLM defaults, and
rankings made at n=150 inverted at n=500. So the review is not optional: it
runs at every analysis window, by subagents, before any candidate is written.

## When

At every analysis window: a screening block is complete (all 4 cells of a
candidate terminal), or at least eight new validator-complete, judge-clean
cells have landed since the last window.

## The tools the reviewers use

- `tools/exp_protocol_cell_read.py <bundle>` — the facts of one cell: cards,
  locks, overrides (current and `relocked_from`), whether each training launch
  came after its lock (trace timestamps, honouring `lock; launch` in one
  command), evaluate.py calls and the sample count of every inspect log
  (~44 KB per sample), RL launches, pitfall signatures (zero-grad, OOM, double
  `<bos>`, stop ids, orphaned vLLM), greedy-config writes, timer readings,
  the last assistant turn.
- `tools/exp_protocol_trace_timeline.py <bundle> [--turns]` — where the hours
  went: every tool call classified (protocol bookkeeping, train launch,
  sampling/eval, waiting on runs, data, environment inspection) with its
  execution time, the model's own generation time, and the first occurrence of
  each stage (first card, first lock, first training launch, first eval,
  first RL, final_model written). `--turns` lists every call over three
  minutes with its trace line.

Both read `results/ptb/<batch>/<cell>/` as the operator commits it:
`solve_parsed.txt.gz` (turn headers `Assistant — turn N | <ts>` /
`User — turn N | <ts>`; Bash calls as `$ cmd` or `"command": …` blocks),
`task/memory/cards/`, `status.json`, `judgement_general.json`.

## How

1. Split the clean cells into groups of three or four, one arm per group, and
   launch one reviewer subagent per group with the brief below. Reviewers read
   only; they write `reports/<cell>.md` under the scratch directory.
2. When all reviewers are done, launch one synthesis subagent over every
   report with the synthesis brief; it writes
   `doc/exp_protocol_iterations/<date>-trace-review-round-NN.md`.
3. Read the synthesis and the best and worst cell of each arm yourself. Turn
   the proposals into candidates: each one item on the allowed surface, traced
   to at least two cells, with a target metric the next block can read and a
   score guardrail; write the specs and the 4-cell screening manifests; update
   `doc/exp_protocol_iterations/directions-ledger.md` (every direction's
   status and the reason it changed; every decision with its alternatives).

## The brief (one reviewer, three or four cells)

Give the reviewer: the task and arms in two sentences; the arm means so far;
the bundle paths with official accuracy; the two tools; the trace format; and
these questions, each to be answered with timestamps and trace line numbers:

1. Timeline — session start and end, hours to the first real training launch,
   the stage sequence with times, hours by category (tool output, corrected by
   reading), time left at the end and the scientist's stated reason for
   stopping, quoted.
2. Recipe decisions and their reasoning — data sources and sizes, few-shot
   prefix handling, SFT hyperparameters, whether on-policy RL or rejection
   sampling was considered; quote every place the scientist reasons about RL,
   RFT, risk or the time budget.
3. Decode config — did it ship a greedy `generation_config`; when; what gain
   it measured; if not, whether it was ever aware.
4. Evaluation practice — the n behind each decision, any inversion at larger
   n, evaluate.py versus its own evaluator, full-test use, paired statistics.
5. Pitfalls — every loss of 0.1 h or more with cause and cost, from the cards
   and from the trace; which traps prior knowledge avoided, with the line that
   shows the knowledge.
6. Protocol interaction (protocol arm) — minutes on cards, check, lock and
   close per card; whether preflight reminders or the pitfalls list were read
   and changed a decision; whether the card format shaped the plan (single
   small interventions, a `falsified_if` that abandoned a direction, the
   comparator rule forcing evals); complaints or workarounds; the cost of the
   bootstrap's first action. (Control arm — what structure the scientist
   imposed on itself instead, and how it waited for runs and ended.)
7. Verdict — the three largest contributors to the score with evidence; the
   one protocol change most likely to have raised this cell; what this cell
   did that the other arm typically did differently.

Each report starts with this header, so the synthesis can tabulate it:

```yaml
cell: p00rNN
arm: protocol | control
accuracy: <official>
hours_used: <h>
hours_to_first_train_launch: <h>
protocol_hours: <h>          # 0 for the control
waiting_hours: <h>
greedy_shipped: yes | no
rl_used: yes | no
rft_tried: yes | no (verdict)
largest_eval_n: <n>
stop_reason: <one line>
top_contributors: [<three short items>]
one_protocol_change: <one line>          # protocol arm
knowledge_to_transfer: [<two or three items>]   # control arm
```

Rules for reviewers: quotes of at most two lines, each with `L<line>` and a
timestamp; distinguish what the scientist said from what the trace shows;
never load a whole trace into context (grep, `sed -n`, python gzip); never
modify the repository.

## The synthesis brief (one subagent, all reports)

Read every `reports/<cell>.md` and the round record's window. Produce:

1. A per-arm table of the header fields (means and per-cell values).
2. The ranked explanations of the score difference, each with the count of
   cells that show it and the strongest quotes.
3. What the protocol cost — hours to first launch, protocol hours, the plan
   shape (number and size of steps, RL attempted or not) — against the control.
4. Proposals: at most four candidates, each exactly one item on the allowed
   surface (`SKILL.md` wording, a `pitfalls.yaml` entry, a preflight check
   with a test, an optional card field), with its source cells, the target
   metric a 4-cell screen would read, and the guardrail; and a list of what the
   protocol should stop doing, if anything, with the cells that show the cost.
5. Open questions the next wave should settle.

The synthesis is a document under `doc/exp_protocol_iterations/`; it is
evidence, not a decision. The decision is yours and goes in the round record
and the ledger.
