| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 67 | - | sft | base_model | gsm8k/main train (chat prompt/completion) | 2e-4 / 2.0 | completed | accuracy 0.02 (n=100, --max-tokens 512) | inconclusive | reject |
| exp-02 | 156 | - | sft | base_model | gsm8k/main train (chat prompt/completion) | 5e-5 / 1.0 | failed | - | inconclusive | iterate |
| exp-03 | 167 | - | sft | base_model | gsm8k/main train (chat prompt/completion) | 5e-5 / 1.0 | completed | accuracy 0.05 (n=20) | inconclusive | reject |
| exp-04 | 222 | - | sft | base_model | gsm8k/main train (plain text) | 5e-5 / 1.0 | completed | - | inconclusive | abandon_line |
| exp-05 | 234 | - | sft | base_model | gsm8k/main train (2000, chat prompt/completion) | 1e-5 / 0.5 | completed | - | inconclusive | abandon_line |
| exp-06 | 249 | - | sft | base_model | gsm8k/main train (2000, plain text + EOS) | 1e-5 / 0.5 | killed | - | inconclusive | iterate |
| exp-07 | 255 | - | sft | base_model | gsm8k/main train (2000, plain text + EOS) | 1e-5 / 0.5 | completed | accuracy 0.1417 (n=120, +0.0084 vs base) | supported | adopt |
| exp-08 | 267 | - | sft | base_model | gsm8k/main train (full, plain text + EOS) | 1e-5 / 1.0 | completed | - (eval run at [273], number never printed) | contradicted | reject |
| exp-09 | 283 | - | other (package) | exp-07 | - | - / - | completed | accuracy 0.15 (n=20) | inconclusive | adopt |
