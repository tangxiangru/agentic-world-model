| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 80 | null | sft | base_model | openai/gsm8k main/train (256) | 5e-5 / 0.1 | failed | none | inconclusive | abandon_line |
| exp-02 | 86 | null | sft | base_model | openai/gsm8k main/train (256) | 5e-5 / 0.1 | completed | none (train_loss 1.107, 6 steps) | inconclusive | abandon_line |
| exp-03 | 90 | null | sft | base_model | openai/gsm8k main/train (full, n not printed) | 1e-4 / 3 (killed in epoch 2) | killed | none | inconclusive | abandon_line |
| exp-04 | 137 | null | merge | exp-03 | none | n/a | completed | accuracy 0.1333 @ n=60 (exp1_ep1_60.json), -0.3167 vs base 0.45 | contradicted | reject |
| exp-05 | 170 | null | sft | base_model | openai/gsm8k main/train (256 of 7473) | 2e-5 / 0.05 | completed | none (train_loss 0.422, 4 steps) | inconclusive | abandon_line |
| exp-06 | 174 | null | sft | base_model | openai/gsm8k main/train (7473) | 2e-5 / 1 | completed | accuracy 0.5833 @ n=60 (exp2_masked_60.json), +0.1333 vs base 0.45 | supported | adopt |
| exp-07 | 199 | null | other (packaging to final_model) | exp-06 | none | n/a | completed | accuracy 0.58 @ n=150 (final_eval_150.json), no same-limit comparator | inconclusive | adopt |
