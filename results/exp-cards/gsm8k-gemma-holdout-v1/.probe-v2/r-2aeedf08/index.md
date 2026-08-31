# r-2aeedf08 — extracted experiment cards

8 launches, all in one turn; the stream carries no timestamps, so `elapsed_h` is null on every card.
Base model: HuggingFaceTB/SmolLM3-3B-Base. Benchmark: gsm8k, 10 h budget, one H100.

| exp-NN | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 69 | null | sft | base_model | gsm8k main train, 128 ex | 1e-4 / 0.1 | completed | none | inconclusive | adopt |
| exp-02 | 75 | null | merge | exp-01 | — | — / — | completed | none (eval @ limit 20 run, value never printed) | inconclusive | reject |
| exp-03 | 100 | null | sft | base_model | gsm8k main train, 512 ex | 1e-4 / 0.5 | completed | none | inconclusive | adopt |
| exp-04 | 103 | null | merge | exp-03 | — | — / — | completed | none (eval @ limit 20 run, value never printed) | inconclusive | reject |
| exp-05 | 118 | null | sft | base_model | gsm8k main train, full split (n never printed) | 1e-4 / 3 | completed | none (scored via its merge, exp-06) | inconclusive | adopt |
| exp-06 | 157 | null | merge | exp-05 | — | — / — | completed | accuracy 0.620, n=100 (also 0.58, n=150) | inconclusive | adopt |
| exp-07 | 173 | null | sft | base_model | gsm8k main + socratic train, full splits (n never printed) | 1e-4 / 1.5 | killed | none | inconclusive | abandon_line |
| exp-08 | 203 | null | other (cp to final_model) | exp-06 | — | — / — | completed | accuracy 0.54, n=50 (baseline 0.18 @ n=50, +0.36) | supported | adopt |

Notes

- Comparator for the whole run: base model, accuracy 0.18 on 50 samples (`baseline_limit50.json`, event [36]). Only exp-08 was measured against it under the same `--limit`, so every other verdict is inconclusive.
- exp-08 is the submitted artifact: `final_model` is a `cp -a` of exp-06's merged directory, scored 0.54 at `--limit 50` (`final_model_eval50.json`).
- `runs/*eval*.json` (sanity 20, quick_fix_eos 20, exp1 100, exp1 150) are not in the workspace snapshot; 0.620 and 0.58 are the agent's own statements at [165] and [196], and the two `--limit 20` evals have no value anywhere in the stream.
