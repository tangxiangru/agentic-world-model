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

## What you may change

| in `skills/wma/` | how |
|---|---|
| `SKILL.md` — level definitions, basis rules, probe playbook, priors | edit; one change per round if you can; a prior that the ledger contradicts is replaced, not softened |
| `verdict.example.json` | keep it valid (`tests/test_wma_skill_files.py`) and representative of the current skill |
| the heuristic backend (`awm/wma/backends.py`) | only to keep the baseline honest; it must stay a set of fixed, stated priors |

You do not change the verdict schema, the scoring rules, or the replayer's
leakage rules from inside a round. Those are measurement; changing them is a
separate decision recorded in `doc/spec/`.

## The loop (offline replay)

1. **Fix the sample set.** The standard set is train side, `--sample 300 --seed 0`
   (`/data2/gangda/hv/wma-replay/round-00/samples.jsonl`). Changing it needs a
   written reason; a new set gets a new baseline run first.
2. **Name the variants.** A variant is a commit of `skills/wma/`; its
   `wma_skill` hash is what the ledger groups by. Baseline is the current
   skill; the heuristic backend is the floor every variant must beat. To compare
   *agents* (backend or model) on the same skill, review the same cards with a
   `--tag` per agent: the ledger separates them by (skill, backend, mode) and the
   tagged files sit side by side.
3. **Run.** One output directory per (variant, backend, model):
   ```bash
   awm wma replay --corpus <corpus> --out <out>/round-NN-<label> --side train --sample 300 --seed 0 \
       --backend claude --model <model> --budget cpu=5,gpu=0,wall=8 [--limit 20]
   awm wma ledger <out>/round-NN-<label>
   ```
   Start every model-backed round with `--limit 20`: check the cost per
   verdict, the wall time, and that the agent writes a valid file, before
   spending on the full set. A round that is cut short is a valid round —
   record it with its `n`.
4. **Read the ledger per level, not just the row.** For each of L0–L3: the hit
   rate against the base rate from round 00 (0.91 / 0.63 / 0.50 at width 0.08 /
   0.43); for L2 the width next to the coverage; for L3 `gpu_h_saved` against
   `gpu_h_wrongly_killed`. A level that moved is where the change acted; a
   level that did not is where the next change should look.
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

- **Never** iterate on the test side of the split. It is run once, on a
  candidate the train side already prefers, to confirm; then never again for
  that candidate.
- **Never** compare rounds that differ in anything but the skill: same sample
  set, same backend, same model, same budget.
- **Never** count a coverage gain that came with a width gain as progress.
- **Never** feed a verdict's own outcome back into the skill as a rule ("card
  r-016546b4/exp-04 fails"); the skill learns kinds, not instances.
- Cost is a result. Record `cost_cpu_min_mean` / `cost_wall_min_mean` and the
  spend per round; a skill that is more accurate and three times slower is a
  trade the record must show.
- A round with no change is a valid round. Record it.

## From offline to online

The online loop is the same with two differences: rollouts come from real
scientist cells on H100 (the verdict is produced at `lock`; once the scientist
closes the card, the ledger scores it — nothing is written back), and dynamic
probes are allowed. Do not go online until the offline
ledger beats the heuristic floor on L0, L1 and L3 and the held-out side
agrees — the thresholds are in `doc/spec/2026-09-01-wma-v1-design.md` §4.4.
