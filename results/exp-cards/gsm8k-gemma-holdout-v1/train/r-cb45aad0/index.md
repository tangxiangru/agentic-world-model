# r-cb45aad0 — reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, one H100.
The digest carries no timestamps, so every `elapsed_h` is null.
All accuracies are the agent's own `evaluate.py` runs; `@50` / `@150` is the `--limit`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 75 | null | sft | base_model | train_data (342,419; MetaMathQA GSM+MATH, GSM8K x3) | 2e-5 / 2.0 | killed | — | inconclusive | abandon_line |
| exp-02 | 115 | null | sft | base_model | train_data_v2 (GSM8K x5 + MetaMathQA GSM + 40k MATH) | 2e-5 / 1.0 | killed | — | inconclusive | adopt |
| exp-03 | 220 | null | sft | exp-02 (checkpoint-2000) | train_data_v2 | 2e-5 / 1.0 (resumed) | completed | 0.440 @50 | inconclusive | adopt |
| exp-04 | 248 | null | other (packaging) | exp-03 | — | — | completed | 0.440 @50 | inconclusive | adopt |
| exp-05 | 281 | null | grpo | exp-03 | openai/gsm8k train prompts | 5e-7 / 1.0 | killed | — | inconclusive | abandon_line |
| exp-06 | 308 | null | sft | base_model | train_data_v3 (GSM8K x8 + MetaMathQA GSM only) | 1e-5 / 3.0 | killed | 0.420 @50 → 0.440 @50 after the stop-token fix | contradicted | reject |
| exp-07 | 475 | null | sft | base_model | train_data_v4 (219,704; eval-format system message) | 1e-5 / 2.0 | killed | 0.380 @50 → 0.440 @50 after the stop-token fix | contradicted | reject |
| exp-08 | 519 | null | decode-config | exp-04 | — | — | completed | 0.560 @50 (0.467 @150) | supported | adopt |
| exp-09 | 562 | null | sft | exp-08 | train_data_v4 | 5e-6 / 1.0 | completed | 0.600 @150 (0.480 @50) | supported | adopt |
| exp-10 | 594 | null | other (packaging) | exp-09 | — | — | completed | 0.600 @150 | inconclusive | adopt |
| exp-11 | 602 | null | sft | exp-09 | train_data_v4 (second pass) | 3e-6 / 1.0 | completed | 0.587 @150 | contradicted | reject |
| exp-12 | 637 | null | sft | exp-09 | train_data_v5 (149,260; GSM8K only, x15 + x5) | 2e-6 / 1.0 | killed | — | inconclusive | abandon_line |

Submission: **exp-10** — `/home/ben/task/final_model` holding the exp-09 weights (0.600 @150). Nothing later overwrote it in the recorded stream.

Not carded (recorded as `provenance.smoke_runs`): [49], [54], [57], [64] on exp-01; [209] on exp-03; [270] on exp-05; [465] on exp-07.
