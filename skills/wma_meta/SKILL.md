---
name: wma_meta
description: Use when iterating the world-model agent itself — running an offline replay round over the historical card corpus (or an online round on H100), reading the ledger, deciding one change to skills/wma, and recording why. Read by the iteration agent only; never by the WMA while it is producing a verdict.
---

# Iterating the world-model agent

You are not the WMA. You run the WMA — one version of `skills/wma/` per
variant — against proposals whose outcomes are known, and you change the
skill based on what the ledger says. The WMA never reads this file.

`skills/wma/` says how to estimate. This file says how to get better at it.
The two accumulate different experience: the WMA's is about proposals, yours
is about which kinds of change to the skill helped and which did not.

The frozen contract for the current stage — sample set, backend, model,
effort, budget, pass counts, candidate-pool and held-out rules — is
`doc/spec/2026-09-02-wma-gsm8k-gemma4b-iteration-basis.md`. This file is the
procedure; that spec holds the numbers.

## What you may change

| in `skills/wma/` | how |
|---|---|
| `SKILL.md` — the procedure, level definitions, basis rules | edit; one change per round if you can |
| `change_types.md` — tiers, the type table, priors with their evidence grade, silent failures, noise floor | edit; a prior the ledger contradicts is replaced, not softened; a new number carries its grade; the manual is part of the skill hash |
| `verdict.example.json` | keep it valid (`tests/test_wma_skill_files.py`) and representative of the current skill |
| the heuristic backend (`awm/wma/backends.py`) | only to keep the baseline honest; it must stay a set of fixed, stated priors |

You do not change the verdict schema, the scoring rules, the replayer's
leakage rules, the sample set, or the prompt in `awm/wma/review.py` from inside
a round. Those are measurement; changing them is a separate decision recorded
in `doc/spec/`.

## The loop (offline replay)

1. **Fix the sample set.** The standard set is the train side restricted to the
   strong-agent runs, `--agents 'claude-(opus-5|fable-5|opus-4-8|opus-4-7)'`,
   no sampling (313 cards; fingerprint in `samples.sha`). Changing it needs a
   written reason; a new set gets a new baseline run first. The filter works on
   the split file's run ids; the sessions never name the agent.
2. **Name the variants.** A variant is a commit of `skills/wma/`; its
   `wma_skill` hash is what the ledger groups by. Baseline is the current
   skill; the heuristic backend is the floor every variant must beat. To compare
   *agents* (backend or model) on the same skill, review the same cards with a
   `--tag` per agent: the ledger separates them by (skill, backend, model,
   effort, mode) and the tagged files sit side by side.
3. **Run.** One output directory per (variant, backend, model, effort, pass):
   ```bash
   awm wma replay --corpus <corpus> --out <out>/round-NN-<label> --side train \
       --agents 'claude-(opus-5|fable-5|opus-4-8|opus-4-7)' \
       --backend claude --model <model> --effort high --budget cpu=5,gpu=0,wall=8,turns=30 [--limit 20] [--jobs 4]
   awm wma ledger <out>/round-NN-<label>            # add --by type or --by family to see where the skill acts
   ```
   Start every model-backed round with `--limit 20`: check the cost per
   verdict, the wall time, that the agent writes a valid file, and that
   `n_leak_suspected` is 0, before spending on the full set. A round that is
   cut short is a valid round — record it with its `n`.
4. **Read the ledger per level, not just the row.** For each of L0–L3: the hit
   rate against the base rate from round 00 (0.91 / 0.63 / 0.50 at width 0.08 /
   0.43). For L0 and L1 also the recall on the cards that failed or produced no
   valid candidate (`L0_recall_failed`, `L1_recall_invalid`) — an agent that
   always says yes already has the base rate; recall is where a skill shows.
   For L2 the coverage next to `L2_width_over_noise` (width divided by the
   noise floor of that card's evaluation size — a wide interval at n=20 is
   honest, at n=1319 it is evasion) and `n_L2_scorable`; for L3 `gpu_h_saved`
   against `gpu_h_wrongly_killed`. Then `--by type`: a change to the manual
   should move the types it touched and nothing else. A level that moved is where the
   change acted; a level that did not is where the next change should look.
5. **Read ten verdicts by hand** — five hits, five misses. The numbers say
   whether; the `basis` and `evidence` say why. A miss with a confident, well-cited
   basis is a skill problem; a miss with an empty basis is a discipline problem.
6. **Decide one change**, traceable to something you read. Write it in the
   record first, then edit the skill.
7. **Record** in `doc/wma_iterations/<date>-round-NN.md` (copy
   `skills/exp_protocol_meta/iteration_record.template.md`; the sections are the
   same). Commit the record and the skill change together.
8. **Promote** only after the held-out side confirms it — see below.

## Rules

- The test side of the split is not for iteration. It is run once, on a
  candidate the train side already prefers, to confirm; then not again for
  that candidate. Only its ledger summary is read — not the verdicts' `basis`
  or `evidence`, which are the failure modes themselves.
- Rounds are compared only when they differ in nothing but the skill: same
  sample set, same backend, same model, same effort, same budget.
- A coverage gain that came with a gain in `L2_width_over_noise` is not progress.
- A verdict's own outcome does not go back into the skill as a rule ("card
  r-016546b4/exp-04 fails"); the skill learns kinds, not instances.
- Cost is a result. Record `cost_usd_mean`, `cost_wall_min_mean` and the spend
  per round; a skill that is more accurate and three times slower is a trade
  the record must show.
- A round with no change is a valid round. Record it.

## The online loop (rounds on H100 cells)

Rollouts are real scientist cells. Since 2026-09-03 the verdict is part of the
lock: `awm exp_protocol lock` asks the sidecar and waits, and the run may not
start before it returns, so a delivered verdict precedes every launch by
construction and the lock file records what happened (`wma.state`). Once the
scientist closes a card the ledger scores it; nothing is written back. Dynamic
probes are allowed. Each harvested cell is
`results/ptb/<batch>/<cell>/`: `metrics.json`, `task/memory/cards/*.yaml` with
`.lock.json` and `.verdict.json`, `task/.wma/` queue records,
`wma_private/*.transcript.jsonl.gz`, `solve_parsed.txt.gz`.

The operator completion trigger is `tools/wma-evolve-hook`. It does not wait
for a PR reviewer: every eight newly validator-clean cells it freezes their
provenance and asks read-only Claude Code Opus 5 at max effort, with the
`ultracode` workflow, to run the delegated trace diagnosis and propose the
next wave. Treat that report as delegated reading, not promotion authority:
the iteration agent still reads the ledger, checks every cited cell and gate,
records the ranked causes, and owns all edits, launches, commits and pushes.
The hook's contract and shared state are in
`doc/reference/wma_evolve_hook.md`.

1. **Evidence window.** At least eight validator-complete cells per arm (or per
   candidate). Slurm `COMPLETED` is not completion; the validator is. In-flight
   snapshots (`<cell>.inflight/`) are for sidecar health and early reading, not
   for the readout.
2. **Delegate the reading; never read sixty traces in the main context.** Read
   the ledger yourself (`awm wma ledger results/ptb/<batch>/*/task --by type`),
   then hand one question each to subagents run in parallel, each writing a file
   under the round directory and returning numbers with citations
   (`w01r05/exp-05`): (a) ledger reproduction by skill hash
   (`tools/wma-rca/verdicts.py`); (b) uptake funnel and timing
   (`tools/wma-rca/uptake.py`, then `timeline.py` on five cells by hand — three
   misses, two hits); (c) score levers (`tools/wma-rca/cells.py`: correlations
   with the final score, arm differences); (d) harm cases — a `no`/`defer` that
   was heeded on a card that would have worked, an over-optimistic L2 that led
   to a wasted run; (e) protocol compliance — `lock.wma.state` distribution,
   verdict-before-launch rate, wait time, timeouts and `--no-wma-wait` reasons.
   Keep the subagent count to the questions that have evidence to read; one
   agent per question, none for restating the ledger.
3. **Rank the causes by evidence weight** from the files, and split them: what
   skill text can fix (calibration, the L3 policy, the form of a suggestion)
   versus what only the protocol or harness can fix (timing, enforcement, what
   the scientist must answer before `close`). Write the ranked list into the
   round record before proposing anything.
4. **Propose candidates, one edit each**: a `skills/wma/` edit *or* a
   protocol/harness edit, never both in one candidate; each names the cause it
   targets, its primary metric, its falsification and its guards (leaks 0, cost
   ≤ 1.5× the baseline, PTB not below the baseline beyond the baseline's spread),
   preregistered in the round spec before any candidate cell runs. Up to one
   wave of parallel candidates at the user's replication (four cells each,
   2026-09-02). Linear commit chain: each candidate is one commit on the
   baseline, the head returns to the baseline; the ledger groups skill
   candidates by `wma_skill`, protocol candidates by `awm.sha`.
5. **Readout per candidate** against the same-wave baseline cells and the
   pooled baseline; promote at most one edit per round; never pool candidates
   post hoc; a second passing edit is retested stacked on the promoted baseline
   next round.
6. **Cost and variance are results.** Report the within-arm SD and the number
   of pairs the claimed effect would need; wait time per cell under the blocking
   lock is GPU idle and goes in the record.

### What Round 01 taught the loop (2026-09-02, 8+8 cells, `tools/wma-rca/`)

- The verdict was outside the decision loop: step 4b was advisory, verdicts
  took 5–9 min, every scientist launched at lock → 21/22 delivered verdicts
  arrived after the launch, 26/59 cards were never reviewed. Fixed in the
  protocol, not the skill: the blocking lock (Round 03 candidate `block`).
- v0.2 never said "don't": L3 `yes` 22/22, so `gpu_h_saved` was 0 by
  construction; its best suggestion (score intermediate checkpoints, 15/22
  verdicts) was taken in 2/8 cells, the control arm's unprompted rate.
- The levers that moved the score were not WMA-shaped: first-SFT quality
  ρ = 0.60, training volume ρ ≈ 0.5 — recipe decisions made before any verdict.
- Calibration (L2 width 6× the floor, misses on both sides) changed no action.
- Within-arm SD 0.03–0.05 against a +0.4 pp delta; n = 8 resolves nothing
  under ~4 pp.

Going online in the first place waited for the offline ledger to beat the
heuristic floor on L0, L1 and L3 with the held-out side agreeing — the
thresholds are in `doc/spec/2026-09-01-wma-v1-design.md` §4.4 and the gates in
`doc/spec/2026-09-02-wma-gsm8k-gemma4b-iteration-basis.md` §十.
