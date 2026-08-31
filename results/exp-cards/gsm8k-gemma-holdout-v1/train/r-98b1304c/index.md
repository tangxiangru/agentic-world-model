# r-98b1304c — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 54 | null | sft (LoRA r=64) | base_model | openai/gsm8k main/train | 2e-4 / 4 | failed | none | inconclusive | iterate |
| exp-02 | 63 | null | sft (LoRA r=64) | base_model | openai/gsm8k main/train | 2e-4 / 4 | failed | none | inconclusive | iterate |
| exp-03 | 74 | null | sft (LoRA r=64) | base_model | openai/gsm8k main/train | 2e-4 / 4 | killed | eval_loss 0.4294 @ step 200 (no accuracy) | inconclusive | adopt |
| exp-04 | 119 | null | merge | exp-03 | none (merge) | null / null | completed | accuracy 0.125 (n=40) | inconclusive | reject |
| exp-05 | 167 | null | sft (LoRA r=32) | base_model | openai/gsm8k main/train | 1e-4 / 2 | killed | eval_loss 0.4747 @ step 50 (no accuracy) | inconclusive | abandon_line |
| exp-06 | 190 | null | sft (LoRA r=32, fewshot_k=3) | base_model | openai/gsm8k main/train | 8e-5 / 1.5 | killed | eval_loss 0.4831 @ step 40 (no accuracy) | inconclusive | abandon_line |
| exp-07 | 209 | null | sft (full-parameter) | base_model | openai/gsm8k main/train | 2e-5 / 1.5 | completed | accuracy 0.200 (n=40), +0.075 vs exp-04 | supported | adopt |
| exp-08 | 233 | null | sft (full-parameter) | base_model | openai/gsm8k main/train | 3e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-09 | 261 | null | other (packaging to final_model) | exp-07 | none (copy) | null / null | completed | accuracy 0.31 (n=100); 0.15 (n=40) | inconclusive | adopt |

The run carries no timestamps, so `elapsed_h` is null on every card.
`exp-09` is the submitted model (`final_model`, a copy of exp-07's `runs/full_run1`).
