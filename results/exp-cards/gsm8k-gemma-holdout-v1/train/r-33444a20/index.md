| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 80 | 0.17 | sft | base_model | meta-math/MetaMathQA (395,000) | 2e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-02 | 84 | 0.21 | sft | base_model | meta-math/MetaMathQA (395,000) | 2e-5 / 1 | completed | accuracy 0.127 (n=150) | inconclusive | adopt |
| exp-03 | 102 | 3.75 | decode-config | exp-02 | none | n/a | completed | accuracy 0.240 (n=150) | supported | reject |
| exp-04 | 120 | 3.88 | sft | base_model | meta-math/MetaMathQA (395,000), eval-format targets | 2e-5 / 1 | completed | accuracy 0.633 (n=150) | supported | adopt |
| exp-05 | 160 | 7.60 | other (packaging) | exp-04 | none | n/a | completed | accuracy 0.613 (n=150) | inconclusive | adopt |
