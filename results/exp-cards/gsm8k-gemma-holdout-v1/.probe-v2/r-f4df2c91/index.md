# reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [20] | 0.02 | other | base_model | - | - / - | completed | 0.090 @ n=100 | inconclusive | reject |
| exp-02 | [35] | 0.09 | sft | base_model | openai/gsm8k main train (n unstated) | 2e-05 / 3 | failed | - | inconclusive | abandon_line |
| exp-03 | [39] | 0.10 | sft | base_model | openai/gsm8k main train (n=7473) | 2e-05 / 3 | failed | - | inconclusive | abandon_line |
| exp-04 | [45] | 0.12 | sft | base_model | openai/gsm8k main train (n=7473) | 2e-05 / 3 | failed | - | inconclusive | abandon_line |
| exp-05 | [51] | 0.13 | sft | base_model | openai/gsm8k main train (n=7473) | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-06 | [69] | 0.88 | sft | base_model | openai/gsm8k main train (n=2000) | 3e-05 / 2 | completed | - | inconclusive | adopt |
| exp-07 | [81] | 0.94 | merge | exp-06 | - | - / - | completed | - | inconclusive | reject |
| exp-08 | [96] | 0.98 | merge | exp-06 | - | - / - | failed | - | inconclusive | abandon_line |
| exp-09 | [102] | 0.99 | merge | exp-06 | - | - / - | completed | 0.040 @ n=100 | contradicted | reject |
| exp-10 | [113] | 1.05 | sft | base_model | openai/gsm8k main train (n=7473) | 0.0001 / 3 | killed | - | inconclusive | abandon_line |
| exp-11 | [124] | 1.23 | sft | base_model | openai/gsm8k main train (n=1000) | 0.0002 / 2 | completed | - | inconclusive | adopt |
| exp-12 | [128] | 1.25 | merge | exp-11 | - | - / - | failed | - | inconclusive | abandon_line |
| exp-13 | [130] | 1.26 | merge | exp-11 | - | - / - | completed | 0.060 @ n=100 | contradicted | reject |
| exp-14 | [144] | 1.36 | other | base_model | - | - / - | completed | 0.120 @ n=100 | supported | adopt |
| exp-15 | [151] | 1.40 | sft | base_model | openai/gsm8k main train (n=7473) | 5e-05 / 1 | completed | - | inconclusive | adopt |
| exp-16 | [153] | 2.09 | merge | exp-15 | - | - / - | completed | 0.030 @ n=100 | contradicted | reject |
| exp-17 | [160] | 2.14 | other | base_model | - | - / - | completed | 0.127 @ n=150 | inconclusive | adopt |
