| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 69 | null | sft | base_model | gsm8k main (128) | 1e-4 / 0.1 | completed | none | inconclusive | adopt |
| exp-02 | 75 | null | merge | exp-01 | - | - / - | completed | none (limit-20 eval ran, value never printed) | inconclusive | reject |
| exp-03 | 100 | null | sft | base_model | gsm8k main (512) | 1e-4 / 0.5 | completed | none | inconclusive | adopt |
| exp-04 | 103 | null | merge | exp-03 | - | - / - | completed | none (limit-20 eval ran, value never printed) | inconclusive | reject |
| exp-05 | 118 | null | sft | base_model | gsm8k main (full) | 1e-4 / 3 | completed | none | inconclusive | adopt |
| exp-06 | 157 | null | merge | exp-05 | - | - / - | completed | accuracy 0.62 @ n=100 (also 0.58 @ n=150) | inconclusive | adopt |
| exp-07 | 173 | null | sft | base_model | gsm8k main+socratic (full) | 1e-4 / 1.5 | killed | none | inconclusive | abandon_line |
| exp-08 | 203 | null | other (packaging) | exp-06 | - | - / - | completed | accuracy 0.54 @ n=50 (base 0.18, +0.36) | supported | adopt |
