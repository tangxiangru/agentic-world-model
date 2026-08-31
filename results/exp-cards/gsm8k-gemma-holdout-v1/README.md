# Experiment cards reconstructed from `gsm8k-gemma-holdout-v1`

**2,030 cards over all 193 runs** (143 train + 50 test), one directory per run
named by an opaque `run_ref`, one card per launch the agent made
(`exp-01.yaml`, `exp-02.yaml`, …), following `card.template.yaml` **in this
directory** — a frozen snapshot of `doc/exp-card-specs/experiment-card.template.yaml`
taken when extraction started (the live template has since gained
world-model-agent fields that past runs cannot fill) — plus the `provenance`
block defined in `doc/exp-card-specs/extraction-protocol.md`.

```text
card.template.yaml               the frozen schema every card follows
train/<run_ref>/exp-NN.yaml      143 runs — Qwen3-4B, Qwen3-1.7B, SmolLM3-3B   (1,591 cards)
train/<run_ref>/index.md         the extractor's per-run summary table
test/<run_ref>/exp-NN.yaml        50 runs — gemma-3-4b-pt, the held-out side    (439 cards)
index.md                         one line per card, all 2,030
coverage.json                    field coverage counts + the checker's flag list
```

`run_ref → run path` lives outside this directory, in
`data/exp-cards/gsm8k-gemma-holdout-v1/sources.json` (gitignored), for audit
only. **Never seed a memory for a gemma run from `test/`** — that is the split's
answer key.

## How they were made

An Opus 5 subagent per run read (a) a digest of the run's event stream —
recipe-, launch- and evaluation-bearing events with their command outputs,
headed `--- [i] turn=T t=+H.HHh act ---`, carrying **no score and no agent
identity** — and (b) the run's `task/` workspace snapshot from the HF release
(scripts, the agent's own eval JSONs; no checkpoints). It wrote one card per
candidate-producing launch under the rules in `extraction-protocol.md`; the
orchestrator reviewed reports and samples, fixed the pipeline three times
(see below), and re-ran the affected runs. Post-processing was deterministic:
strip template drift, join `outcome.official_accuracy` from the pinned
catalogue onto each run's adopted card, lint for identity, build the index.

Digests are rendered by `tools/render_card_digests.py`; prompts by
`tools/exp_cards_prompts.py`; the post-pass is `tools/exp_cards_post.py`
(all on `main`, commit `00c6823`).

## What the corpus looks like

| | |
|---|---|
| cards per run | min 1, median 9, max 66 |
| `method.family` | sft 1000 · merge 291 · other 269 (mostly packaging) · decode-config 168 · grpo 139 · rft 102 · distill 33 · dpo 28 |
| `conclusion.decision` | adopt 855 · reject 731 · abandon_line 370 · iterate 74 |
| `conclusion.verdict` | inconclusive 1191 · contradicted 487 · supported 352 |
| hypothesis stated by the agent | 1221 cards (60 %); reconstructed or absent on the rest |
| `elapsed_h` filled | 1469 cards (the 130 runs whose harness recorded timestamps) |
| smoke/dry runs folded into `provenance.smoke_runs` | 539 |
| official score joined | 193 runs (one adopted card each) |

Field coverage (non-empty): `problem.statement` 75 %, `hypothesis.claim` 65 %,
`evaluation.comparator` 82 %, `result.measurements` 67 %, `setup.data` 64 %,
`problem.failure_examples` 9 %, `hypothesis.falsified_if` 6 %. The low ones are
honest: past agents almost never wrote a falsification condition or printed
failing items, and the protocol forbids inventing either.

## Rules the cards were written under

- **Empty means the trajectory did not say.** `null` / `[]` are gaps, not
  defaults. `provenance.stated_by_agent` says whether `problem` and
  `hypothesis` are the agent's own words.
- **No identity.** No card names the agent model, harness, experiment or run.
  The digests carried none of these; the lint over the serialised YAML finds
  zero hits (whole-word; `precursor` and `openai/gsm8k` are not hits).
- **No official score at extraction time.** Joined afterwards, adopted card only.
- **Paths are the run's own** (`/home/ben/task/...`). The files still exist in
  the workspace snapshot at `data/traj/raw/posttrainbench/<run>/task/` when the
  release kept them; `provenance.snapshot_files` lists the ones each card cites.
- **Smoke tests are not cards**; they ride on the next real launch as
  `provenance.smoke_runs`.

## Known limitations

- **18 checker flags** (`coverage.json → problems`): all are launches written
  as heredocs or compound commands (`python - <<PY …`, `sleep 15 && …`,
  `cat > … <<EOF`) whose flattened argv cannot be matched as a substring of the
  digest block. Every one still cites the right `launch_i`; the flag is a
  limitation of the text check, not evidence of a wrong card.
- **6 runs hit the digest size cap** (280k chars) and lost their first hours;
  their cards say so in `provenance.unresolved` and early parents fall back to
  `base_model` / `null`.
- **Config-only candidates** (a hand-edited `generation_config.json`, an EOS
  patch) often have no command event in the stream. They are carded on the
  first evaluation that exercised the changed config, with the gap recorded.
- **Workspace snapshots are uneven**: many runs kept only scripts, so
  `result.measurements[].path` frequently names an eval JSON that existed in
  the run but is not in `task/`; the value then comes from the printed eval
  summary in the stream, which the card cites by event index.
- One run (`train/r-bda2d6c1`) folded packaging steps into training cards
  rather than carding them; its adopted checkpoint is still identified.

## Pipeline fixes made during extraction (worth committing on `main`)

1. `awm/analysis/recipe.py` — `select()` read a tool result's text from
   `args`/`summary`; converters put it in `text`, so **no result had ever
   reached a digest**. Fixed; results (eval printouts, losses, tracebacks) now
   appear.
2. `awm/traj/convert_codex.py` — item ids restart at `item_0` per thread, so a
   re-prompted run's later thread overwrote the argv of an earlier thread's
   same-numbered command. Fixed by scoping ids per thread (8 runs affected).
3. The recipe filter missed `train_v3.py`-style launches and config scripts;
   the card digest widens it (`tools/render_card_digests.py`).

## Proposed fields (beyond the frozen template)

Already used in every card: the `provenance` block (`launch_i`,
`stated_by_agent`, `anchors`, `snapshot_files`, `smoke_runs`, `unresolved`).
Worth adding to the live template: `setup.data[].repeats` (the "×5" decision
has nowhere else to go) and `result.duration_h` / `result.steps` as typed
fields instead of prose in `training_summary.notes`.
