# r-c0173ea9 — train cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 59 | 0.06 | sft | base_model | gsm8k train (7473) | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-02 | 74 | 0.08 | sft | base_model | gsm8k train (7473) | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-03 | 83 | 0.09 | sft | base_model | gsm8k train (7473) | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-04 | 92 | 0.10 | sft | base_model | gsm8k train (7473) | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-05 | 98 | 0.11 | sft | base_model | gsm8k train (7473) | 2e-4 / 3 | killed (step 570/2805) | none | inconclusive | adopt |
| exp-06 | 137 | 0.39 | sft | exp-05 | gsm8k train (7473) | 2e-4 / 2 | completed (train_loss 0.436, 3738 steps) | none | inconclusive | adopt |
| exp-07 | 156 | 1.21 | merge | exp-06 | — | — | completed | none | inconclusive | adopt |
| exp-08 | 161 | 1.23 | other (packaging) | exp-07 | — | — | completed | none (eval launched at [165] printed no score) | inconclusive | adopt |
