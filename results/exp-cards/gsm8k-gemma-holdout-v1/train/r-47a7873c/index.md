# r-47a7873c — gsm8k, Qwen/Qwen3-4B-Base, 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 56 | null | sft | base_model | gsm8k_train_formatted.json (7473) | 2e-5 / 3 | failed | none | inconclusive | abandon_line |
| exp-02 | 62 | null | sft | base_model | gsm8k_train_formatted.json (7473) | 2e-5 / 2 | completed | accuracy 0.320 @ n=50 (metrics_50.json) | inconclusive | reject |
| exp-03 | 110 | null | sft | base_model | gsm8k_train_enhanced.json (7473) | 1e-5 / 3 | completed | none (eval crashed) | inconclusive | adopt |
| exp-04 | 131 | null | other | exp-03 | none | null / null | completed | none | inconclusive | adopt |

Notes: the digest carries no event timestamps, so `elapsed_h` is null on every
card. exp-04 is the packaging step whose output, /home/ben/task/final_model, is
the submission; it holds exp-03's weights, which were never scored.
