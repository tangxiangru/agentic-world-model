| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 20 | 0.02 | other | base_model | - | - | completed | accuracy 0.090 (n=100) | inconclusive | reject |
| exp-02 | 51 | 0.13 | sft | base_model | gsm8k main train (7473) | 2e-5 / 3 | killed | - | inconclusive | abandon_line |
| exp-03 | 69 | 0.88 | sft | base_model | gsm8k main train (2000) | 3e-5 / 2 | completed | - | inconclusive | adopt |
| exp-04 | 81 | 0.94 | merge | exp-03 | - | - | completed | - | inconclusive | abandon_line |
| exp-05 | 102 | 0.99 | merge | exp-03 | - | - | completed | accuracy 0.040 (n=100) | contradicted | reject |
| exp-06 | 113 | 1.05 | sft | base_model | gsm8k main train (7473) | 1e-4 / 3 | killed | - | inconclusive | abandon_line |
| exp-07 | 124 | 1.23 | sft | base_model | gsm8k main train (1000) | 2e-4 / 2 | completed | - | inconclusive | adopt |
| exp-08 | 130 | 1.26 | merge | exp-07 | - | - | completed | accuracy 0.060 (n=100) | contradicted | reject |
| exp-09 | 144 | 1.36 | other | base_model | - | - | completed | accuracy 0.120 (n=100) | supported | reject |
| exp-10 | 151 | 1.40 | sft | base_model | gsm8k main train (7473) | 5e-5 / 1 | completed | - | inconclusive | adopt |
| exp-11 | 153 | 2.09 | merge | exp-10 | - | - | completed | accuracy 0.030 (n=100) | contradicted | reject |
| exp-12 | 160 | 2.14 | other | base_model | - | - | completed | accuracy 0.127 (n=150; 0.107 on a rerun) | inconclusive | adopt |
