| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 88 | null | sft | base_model | gsm8k train (file not named) | 1.5e-4 / 4 | killed | none (eval_loss 0.461 @ epoch 1) | inconclusive | adopt |
| exp-02 | 186 | null | merge | exp-01 | - | - / - | completed | accuracy 0.05 (n=20) | inconclusive | reject |
| exp-03 | 187 | null | merge | exp-01 | - | - / - | completed | none | inconclusive | abandon_line |
| exp-04 | 249 | null | sft | base_model | gsm8k train (file not named) | 2e-5 / 1 | completed | none (eval_loss ~0.466) | inconclusive | adopt |
| exp-05 | 267 | null | merge | exp-04 | - | - / - | completed | accuracy 0.10 (n=20) | inconclusive | reject |
| exp-06 | 289 | null | sft | base_model | gsm8k train (file not named) | 1e-6 / 1 step | completed | none | inconclusive | adopt |
| exp-07 | 295 | null | merge | exp-06 | - | - / - | completed | accuracy 0.467 (n=120) | inconclusive | adopt |
| exp-08 | 323 | null | other (packaging) | exp-07 | - | - / - | completed | none (final eval killed) | inconclusive | adopt |
