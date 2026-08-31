| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 102 | 0.45 | sft | base_model | meta-math/MetaMathQA train (full, <=2048 tok) | 2e-5 / 1 | failed (CUDA OOM at step 158/6172) | none | inconclusive | abandon_line |
| exp-02 | 116 | 0.65 | sft | base_model | meta-math/MetaMathQA train (394,994 after <=2048 tok filter) | 2e-5 / 1 | completed (6172 steps, 6.72 h) | accuracy 0.600 (n=10, official --limit 10) | inconclusive | adopt |
