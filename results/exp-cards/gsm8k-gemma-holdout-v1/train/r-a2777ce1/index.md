| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [28] | 0.08 | sft | base_model | gsm8k:7473 + MetaMathQA:50000 | 2e-05 / 3 | failed | - | inconclusive | abandon_line |
| exp-02 | [46] | 0.12 | sft | base_model | gsm8k:7473 + MetaMathQA:50000 | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-03 | [70] | 0.32 | sft | base_model | gsm8k:7473 + MetaMathQA:30000 | 5e-05 / 2 | killed | - | inconclusive | abandon_line |
| exp-04 | [98] | 0.54 | sft | base_model | gsm8k:7473 + MetaMathQA:20000 | 5e-05 / 2 | killed | - | inconclusive | abandon_line |
| exp-05 | [120] | 0.74 | sft | base_model | gsm8k:7473 | 5e-05 / 1 | completed | 0.340 @n=50 | supported | adopt |
| exp-06 | [140] | 0.95 | sft | base_model | gsm8k:7473 + MetaMathQA:20000 | 5e-05 / 2 | completed | 0.327 @n=150 | inconclusive | adopt |
| exp-07 | [162] | 1.50 | other | exp-06 | - | - | completed | - | inconclusive | adopt |
| exp-08 | [164] | 1.50 | sft | base_model | gsm8k:7473 | 0.0001 / 3 | completed | 0.507 @n=150 | supported | adopt |
| exp-09 | [176] | 1.73 | other | exp-08 | - | - | completed | - | inconclusive | adopt |
| exp-10 | [178] | 1.74 | sft | base_model | gsm8k:7473 + MetaMathQA:30000 | 0.0001 / 3 | completed | 0.453 @n=150 | inconclusive | reject |
| exp-11 | [210] | 2.70 | other | exp-08 | - | - | completed | 0.573 @n=150 | inconclusive | adopt |
| exp-12 | [216] | 2.74 | sft | base_model | gsm8k:7473 | 0.00015 / 4 | completed | 0.333 @n=150 | contradicted | reject |
| exp-13 | [236] | 3.12 | sft | base_model | gsm8k:7473 | 0.0002 / 3 | completed | 0.100 @n=150 | contradicted | reject |
