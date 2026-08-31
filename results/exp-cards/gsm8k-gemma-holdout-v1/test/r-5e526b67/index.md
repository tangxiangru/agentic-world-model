| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 81 | null | sft | base_model | openai/gsm8k train | 1e-4 / 1.0 | completed | accuracy 0.3375 @ n=80 | inconclusive | reject |
| exp-02 | 130 | null | sft | base_model | openai/gsm8k train | 8e-5 / 2.0 | completed | accuracy 0.425 @ n=80 (+0.0875 vs exp-01) | supported | adopt |
| exp-03 | 179 | null | decode-config | exp-02 | none | null | completed | accuracy 0.475 @ n=80 (+0.05 vs exp-02) | supported | adopt |
| exp-04 | 190 | null | sft | base_model | openai/gsm8k train | 6e-5 / 3.0 | killed | none | inconclusive | abandon_line |
| exp-05 | 225 | null | decode-config | exp-03 | none | null | completed | accuracy 0.447 @ n=150 | inconclusive | adopt |
| exp-06 | 234 | null | other (packaging) | exp-05 | none | null | completed | accuracy 0.35 @ n=40 (+0.25 vs base_model) | inconclusive | adopt |
