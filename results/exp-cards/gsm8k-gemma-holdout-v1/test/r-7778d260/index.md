# Extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 48 | 0.13 | sft | base_model | meta-math/MetaMathQA 100k | 2e-4 / 2000 steps (epochs unset) | killed | none | inconclusive | abandon_line |
| exp-02 | 86 | 0.31 | sft | base_model | meta-math/MetaMathQA 100k | 2e-4 / 2000 steps = 0.64 epoch | completed | accuracy 0.500 (n=150, results.json) | inconclusive | adopt |
| exp-03 | 112 | 2.14 | other (packaging) | exp-02 | none | n/a | completed | accuracy 0.500 (n=150, results.json) | inconclusive | adopt |
