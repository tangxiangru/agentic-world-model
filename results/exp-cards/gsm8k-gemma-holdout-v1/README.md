# Experiment cards reconstructed from `gsm8k-gemma-holdout-v1`

One directory per run, named by an opaque `run_ref`; inside, one card per
launch the agent made (`exp-01.yaml`, `exp-02.yaml`, …) following
`card.template.yaml` **in this directory** — a frozen snapshot of
`doc/exp-card-specs/experiment-card.template.yaml` taken when extraction
started (the live template has since gained world-model-agent fields that
past runs cannot fill) — plus the `provenance` block defined in
`doc/exp-card-specs/extraction-protocol.md`.

```text
train/<run_ref>/exp-NN.yaml     143 runs — Qwen3-4B, Qwen3-1.7B, SmolLM3-3B
test/<run_ref>/exp-NN.yaml       50 runs — gemma-3-4b-pt (held out; never seed a memory from these)
index.md                         one line per card
sources.json                     run_ref -> run path, for audit only
```

Rules the cards were written under:

- **Empty means the trajectory did not say.** `null` / `[]` are honest gaps,
  not defaults. `provenance.stated_by_agent` says whether `problem` and
  `hypothesis` are the agent's own words or reconstructed from the launch.
- **No identity.** No card names the agent model, harness, experiment, or run.
  The digests the extractor read carried none of these, and the cards were
  linted for them afterwards.
- **No official score at extraction time.** `outcome.official_accuracy` was
  joined afterwards onto the single adopted card per run, from the catalogue.
- **Paths are the run's own** (`/home/ben/task/...`). The files still exist in
  the workspace snapshot at `data/traj/raw/posttrainbench/<run>/task/`;
  `provenance.snapshot_files` lists the ones each card cites.

Inputs: digests and workspace links under `data/exp-cards/gsm8k-gemma-holdout-v1/`
(gitignored), rendered by `tools/render_card_digests.py`.
