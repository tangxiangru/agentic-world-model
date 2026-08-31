# r-db1eeb72 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 86 | null | sft | base_model | openai/gsm8k train (minus 256 val) | 1.5e-5 / 3.0 | completed | 0.703 acc, n=128 held-out train split (runs/gsm8k_ft_v1/holdout_128.json) | inconclusive | adopt |
| exp-02 | 181 | null | sft | exp-01 | openai/gsm8k train, 5 solved examples in system prompt | 5e-6 / 1.0 | completed | 0.350 acc, n=20 benchmark slice (runs/gsm8k_ft_v2_fs5/eval_20.json) | inconclusive | adopt |
| exp-03 | 252 | null | sft | exp-02 | openai/gsm8k + EleutherAI/asdiv + mwpt5/MAWPS + cq01/mawps-asdiv-a_svamp | 2e-6 / 0.7 | completed | 0.250 acc, n=20 benchmark slice (runs/gsm8k_mix_v1/eval_20.json), -0.10 vs exp-02 | contradicted | reject |
| exp-04 | 274 | null | other (packaging: cp to final_model) | exp-02 | none | null / null | completed | 0.350 acc, n=20 benchmark slice (final_model/eval_20.json), +0.00 vs exp-02 | inconclusive | adopt |
