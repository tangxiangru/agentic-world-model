| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 34 | 0.05 | sft | base_model | gsm8k/main train (in-process) | 2e-5 / 3 | failed | - | inconclusive | abandon_line |
| exp-02 | 44 | 0.07 | sft | base_model | gsm8k/main train (in-process) | 2e-5 / 3 | killed | - | inconclusive | abandon_line |
| exp-03 | 48 | 0.40 | sft | base_model | gsm8k/main train (in-process) | 2e-5 / 2 | completed | accuracy 0.14 (n=50) | inconclusive | adopt |
| exp-04 | 56 | 0.70 | other (package) | exp-03 | - | - / - | completed | - | inconclusive | adopt |
