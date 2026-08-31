# r-27c73665 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 7 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 24351 | 0.36 | sft | base_model | sft_round1.jsonl (gsm8k train + MetaMathQA GSM_* + Orca-Math, ~167k) | 2e-5 / 1.0 | killed | none (killed in packing, no training step) | inconclusive | abandon_line |
| exp-02 | 29086 | 0.44 | sft | base_model | sft_round1.jsonl (same file) | 2e-5 / 1.0 | killed | none (killed in tokenization, no training step) | inconclusive | abandon_line |
| exp-03 | 32052 | 0.82 | sft | base_model | sft_round1.jsonl (same file) | 2e-5 / 1.0 | completed | accuracy 0.627, n=150 (checkpoint-1600); 0.6194 on the full 1319 | inconclusive | adopt |
| exp-04 | 43357 | 4.89 | sft (round 2: rejection samples + replay) | exp-03 | sft_round2.jsonl (46.9k RS from checkpoint-1600 + 39.7k replay) | 1e-5 / 70M-token budget | failed | none (crashed at checkpoint save) | inconclusive | abandon_line |
| exp-05 | 45393 | 5.75 | sft (round 2: rejection samples + replay) | exp-03 | sft_round2.jsonl (same file) | 1e-5 / 70M-token budget | completed | accuracy 0.673, n=150 (checkpoint-950); 0.6657 on the full 1319 (final) | supported | adopt |
| exp-06 | 46893 | 7.99 | other (packaging checkpoint-950 → final_model, greedy decode config) | exp-05 | — | — / — | completed | accuracy 0.6596, n=1319 (final_full.json) | contradicted | reject |
| exp-07 | 47771 | 8.28 | other (packaging ckpts/sft_r2/final → final_model, greedy decode config) | exp-05 | — | — / — | completed | accuracy 0.6657, n=1319; 0.693, n=150 on the exact submitted directory | supported | adopt |

Notes

- The submitted artifact is exp-07: `final_model` rebuilt from `ckpts/sft_r2/final` at
  [47771], the last write to that path, matching the closing inventory at [48714].
  exp-06 wrote the same path from `checkpoint-950` and was reversed 20 minutes later.
- `elapsed_h` is the `t=+H.HHh` of each launch block; the digest carries timestamps
  throughout.
- Three cards (exp-01, exp-02, exp-04) are launches the agent meant as real and then
  lost — two killed in data preparation before any optimizer step, one crashed at its
  first checkpoint save. Only the packing/tokenization/serialisation code changed
  between them and their successors; the recipe did not.
- Smoke runs are not cards: the 2k-example, 40-step training at [21210] and the two
  limit-8 evals of that checkpoint at [22674] (vLLM would not start) and [23620] (62.5%)
  are recorded on exp-01 as `provenance.smoke_runs`.
- Two launches that produced no candidate are deliberately not cards: the limit-50
  baseline eval of the base model at [6393] (0.08, evidence for exp-01's problem) and
  the rejection-sampling generation at [42635] (it builds `data/rs_main.jsonl` and
  `data/rs_mm.jsonl`, recorded as `setup.data[].build_command` on exp-04 and exp-05).
- No same-protocol baseline exists for round 1: the base model was measured only at
  `--limit 50` while every checkpoint was measured at `--limit 150`, so exp-03 is
  `inconclusive` despite 0.08 → 0.627. From exp-05 on, comparators are same-limit and
  the verdicts are decided.
- The agent stated a problem before every launch but never wrote an expectation in
  hypothesis form, so `provenance.stated_by_agent.hypothesis` is `false` on all seven
  cards and each `hypothesis.claim` is a minimal reconstruction from the launch and the
  agent's quoted words.
- Candidate selection was done on the official test set (six candidates at limit 150/300,
  three on the full 1319 items) rather than a held-out dev split of the agent's own; the
  agent flags this risk itself in `RESULTS.md`.
- Data counts (167k round 1, 86.5k round 2, 46.9k rejection samples) are the agent's
  rounded figures from the stream and `RESULTS.md`; the scripts' stdout is filtered out
  of the digest, so every `n_examples` is `null`.
