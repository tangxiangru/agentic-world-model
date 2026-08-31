# r-37da5218 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 394 | 1.11 | sft | base_model | sft_main.jsonl — OpenMathInstruct-2 gsm+math slice + gsm8k train (agent: ~242K recs / 140M tok) | 2e-5 / 1.0 | completed | accuracy 0.740, n=150 (eval_main150.json) | inconclusive | adopt |
| exp-02 | 729 | 4.30 | sft | exp-01 | round2_mix.jsonl — 200,000 fresh OMI-2 + 14,236 RFT + 440 hard-original copies (214,676) | 1e-5 / 1.0 | completed | accuracy 0.727, n=150 (eval_round2_150.json) | inconclusive | reject |
| exp-03 | 824 | 6.47 | merge | exp-01 (+ exp-02 as `--b`) | — (weight average, alpha 0.5 default) | — / — | completed | none (never evaluated) | inconclusive | abandon_line |
| exp-04 | 873 | 7.38 | rft | exp-01 | star_mix.jsonl — 72,437 verified STaR self-solutions + 14,236 RFT (86,673) | 6e-6 / 1.0 | completed | accuracy 0.740, n=150 (eval_star150.json) | inconclusive | adopt |
| exp-05 | 919 | 7.98 | merge | exp-01 (+ exp-04 as `--b`) | — (weight average, alpha 0.5 default) | — / — | completed | accuracy 0.747, n=150 (eval_soupms150.json) | inconclusive | adopt |
| exp-06 | 964 | 8.09 | other (package to final_model, greedy generation_config) | exp-05 | — | — / — | completed | accuracy 0.740, n=150 (eval_final150.json; eval_final_defaults.json identical) | inconclusive | adopt |
