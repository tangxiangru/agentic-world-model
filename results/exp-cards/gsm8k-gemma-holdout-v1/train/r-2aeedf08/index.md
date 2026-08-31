| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 118 | - | sft | base_model | gsm8k/main train (7473) | 1e-4 / 3 | completed | - | inconclusive | adopt |
| exp-02 | 157 | - | merge | exp-01 | - | - / - | completed | accuracy 0.62 (n=100) | inconclusive | adopt |
| exp-03 | 173 | - | sft | base_model | gsm8k/main+socratic train | 1e-4 / 1.5 | killed | - | inconclusive | abandon_line |
| exp-04 | 203 | - | other (package) | exp-02 | - | - / - | completed | accuracy 0.54 (n=50) | supported | adopt |
