# World-model agent

> **Status:** design in force since 2026-09-01 (supersedes the ping/yield protocol) · **Scope:** PostTrainBench scientist runs

## Two agents, one verb

Two Claude Code sessions share one sandbox and talk to each other with
`SendMessage` (verified 2026-09-01: a harness-started headless session appears
in the other's `ListAgents` within a second and can send and receive messages).

- The **research scientist** does the PostTrainBench task as it always has:
  picks data, writes the trainer, trains, evaluates its own checkpoints with
  the benchmark grader, ships `final_model/`. It owns the GPU and every
  decision. Its prompt is PTB's `prompt.txt` plus one section saying a
  world-model agent exists and can be consulted.
- The **world-model agent (WMA)** manages the record of past experiments — the
  corpus it is allowed to read and the experiments of this session — and
  answers one request: **consult**. It never trains, never evaluates, never
  decides, and never speaks unless spoken to.

The scientist consults at any time — with a plan, with a plan and new results,
with a question — and the answer always has the same shape:

| field | content |
|---|---|
| `card` | the WMA's structured understanding of the experiment (`awm-experiment-card-v1`, sections 1–4, plus reported `results` and `gaps` — what it could not determine, as questions) |
| `verdict` | `SURE_WONT_WORK` / `SURE_WILL_WORK` / `CANNOT_DECIDE`, a confidence, a predicted final delta vs the parent with a spread, and the past experiments it rests on (cited) |
| `eval_plan` | when it would be informative for the scientist to evaluate a checkpoint, with what protocol, and the number to beat there |
| `suggestion` | `TERMINATE` / `KEEP_RUNNING` (what information would settle it) / `ADJUST` (one change and the run that motivates it) |
| `reasons` | short claims, each citing a path |

On a re-consult the same shape comes back updated: results verified and added
to the card, verdict tightened or flipped, plan advanced, suggestion moved.
When the scientist ships, the WMA records the outcome next to its predictions.

## Discipline that makes the shape mean something

- `SURE_*` requires confidence ≥ 0.75 **and** citations; otherwise `CANNOT_DECIDE`. Fixed across arms so they differ in evidence, not nerve.
- Every claim cites a path under the allowed roots; `awm wm log` lints citations and downgrades an uncited `SURE_*` to `CANNOT_DECIDE`.
- The card is the WMA's record; the scientist never sees a template. Gaps are questions in the answer, not blockers.
- The eval plan is advice about *when to look*; with nothing comparable it falls back to 25/50/75 % of the planned steps.
- Every consult — request, response, timestamp, verdict, prediction — is appended to `wm/consults.jsonl`. That ledger is how the world model is scored afterwards.

## Arms: what the WMA may read

| arm | evidence | how |
|---|---|---|
| `null` | nothing | card + default eval plan + `CANNOT_DECIDE`; the control for the mechanism itself |
| `retrieval` | extracted experiment cards in memory | `awm wm search` (deterministic nearest cards with outcomes and curves) |
| `traj` | the raw prior runs (`INDEX.md`, complete `solve_out.txt` trajectories, scores, durations) at `prior_runs_root`; prior scripts and task snapshots are not exposed | reads them itself |
| `llm` | both | both |

The WMA's own model is pinned (`WMA_MODEL`, default Opus 4.8) so only the
scientist model varies on that axis. `wm/config.json` names the arm and roots.

## Pieces

| piece | where |
|---|---|
| the WMA's role, response contract, standing order | `wma/CLAUDE.md` |
| the consult procedure it follows | `wma/.claude/skills/consult/SKILL.md` |
| its card schema | `wma/card.template.yaml` (copy of `doc/exp-card-specs/experiment-card.template.yaml`) |
| response schema, validator, citation lint, default eval plan, ledger, outcomes | `awm/wm/consult.py` |
| card drafting from the scientist's words + workspace | `awm/wm/intake.py` |
| memory: precedents, curves, seeding from `results/exp-cards/` | `awm/wm/memory.py` |
| toolbelt CLI: `awm wm init \| draft-card \| search \| eval-plan \| read-eval \| log \| outcome \| status \| memory` | `awm/cli.py` |
| the scientist's one section | `input/wma_section.md` → rendered into `input/instruction.md` and the study prompts by `rollout/build_prompts.py` |
| the two-session cell | `rollout/agents/claude_wm/solve.sh` |
| tests | `tests/test_wm.py`, `tests/test_rollout.py` |

## The study

C1: raw prior runs mounted, no WMA. C2: raw prior runs + a `traj` WMA. C3: a
`retrieval` WMA over the extracted cards, no raw runs. Two prior-information
versions (143 train-side runs, or all 193 with the 50 gemma runs; scores and
identity visible). Primary outcome: PTB's official accuracy of `final_model/`.
Secondary, from the ledger: prediction accuracy and calibration at each
consult, decision value (did the scientist follow the suggestion, and were the
terminated runs the bad ones), and how often the scientist consulted at all.

## Open

- Keep-alive of a headless WMA session over ten hours (standing order: sleep-and-drain loop; a pty fallback if `-p` proves fragile) — settle with one smoke cell.
- Whether a `null` WMA should still shape *when* the scientist evaluates (default schedule) or say nothing at all.
- Whether to add a pushed-update arm later (the WMA volunteers once per checkpoint); the single-`consult` design deliberately excludes it.

## History

The first design (2026-08-30) put a deterministic runtime between the two: the
scientist wrote cards, the runtime froze an evaluation contract, took the GPU
at agreed checkpoints via a hook, evaluated, pinged, sealed. It was replaced on
2026-09-01 because it made the scientist learn a protocol the corpus agents
never had (confounding the ablation) and because the GPU hand-off was the most
fragile part of the system. Two peer sessions and one verb need neither.
