| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 55 | 0.26 | sft | base_model | openai/gsm8k train (7473) | null/null | failed | none | inconclusive | abandon_line |
| exp-02 | 65 | 0.28 | sft | base_model | openai/gsm8k train (7473) | 2e-4/3 | killed | none | inconclusive | abandon_line |
| exp-03 | 67 | 0.45 | sft | base_model | openai/gsm8k train (7473) | 2e-4/3 | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-04 | 104 | 1.02 | merge | exp-03 | none | n/a | completed | accuracy 0.120 (n=50, vs base 0.140) | contradicted | reject |
| exp-05 | 124 | 1.13 | sft | base_model | openai/gsm8k train (7473), markers corrected | 2e-4/3 | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-06 | 136 | 1.65 | merge | exp-05 | none | n/a | completed | accuracy 0.200 (n=50, vs base 0.140); 0.247 (n=150) | supported | adopt |
| exp-07 | 148 | 1.77 | sft | base_model | openai/gsm8k train (7473), markers corrected | 3e-4/5 | completed | none (never evaluated) | inconclusive | abandon_line |
