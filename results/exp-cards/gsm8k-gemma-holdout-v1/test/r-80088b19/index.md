# r-80088b19 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100 80GB
The digest carries no timestamps, so every card has `elapsed_h: null`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 39 | null | sft (LoRA r=128) | base_model | gsm8k train (7473) | 2e-4 / 4 | failed | — | inconclusive | iterate |
| exp-02 | 45 | null | sft (LoRA r=128) | base_model | gsm8k train (7473) | 2e-4 / 4 | completed | 0.507 @ n=150 (eval_result_v1_full.json) | inconclusive | adopt |
| exp-03 | 101 | null | sft (LoRA r=256) | base_model | gsm8k train x2 (7473) + orca-math (20000) | 1.5e-4 / 3 | killed | — | inconclusive | abandon_line |
| exp-04 | 134 | null | sft (full FT) | base_model | gsm8k train x3 (7473) + MetaMathQA MATH (20000) | 2e-5 / 3 | killed | — | inconclusive | abandon_line |
| exp-05 | 155 | null | sft (full FT) | base_model | gsm8k train (7473) | 3e-5 / 4 | completed | 0.000 @ n=50 (eval_result_v4.json) | contradicted | reject |
| exp-06 | 191 | null | sft (LoRA r=64, continual) | exp-02 | gsm8k train x2 (7473) + MetaMathQA MATH (7500) | 5e-5 / 2 | completed | 0.380 @ n=50 (eval_result_v5.json) | inconclusive | reject |

Adopted / submitted: **exp-02** — its merged LoRA checkpoint is what sits in `final_model/` at the end of the run (overwritten by exp-05 at [155], restored from `final_model_v1_backup` at [174], re-verified at [204]-[207]).
