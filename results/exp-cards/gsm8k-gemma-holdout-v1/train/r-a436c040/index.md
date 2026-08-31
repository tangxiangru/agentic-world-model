| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 39 | null | sft | base_model | openai/gsm8k train, 3000 | 2e-4 / 1.0 | failed | - | inconclusive | iterate |
| exp-02 | 45 | null | sft | base_model | openai/gsm8k train, 3000 | 2e-4 / 1.0 | failed | - | inconclusive | iterate |
| exp-03 | 51 | null | sft | base_model | openai/gsm8k train, 3000 | 2e-4 / 1.0 | completed | - | inconclusive | adopt |
| exp-04 | 59 | null | merge | exp-03 | - | - | completed | accuracy 0.05, limit 40 | contradicted | reject |
| exp-05 | 91 | null | sft | base_model | openai/gsm8k train, 5000 (4750 after val) | 1.5e-4 / 1.0 | completed | - | inconclusive | adopt |
| exp-06 | 97 | null | merge | exp-05 | - | - | completed | accuracy 0.350, limit 40 | supported | reject |
| exp-07 | 109 | null | sft | base_model | openai/gsm8k train, full split minus 256 val | 1e-4 / 2.0 | completed | - | inconclusive | adopt |
| exp-08 | 124 | null | merge | exp-07 | - | - | completed | accuracy 0.425, limit 40 | supported | adopt |
| exp-09 | 133 | null | sft | base_model | openai/gsm8k train, full split minus 256 val | 1e-4 / 2.0 | killed | - | inconclusive | abandon_line |
| exp-10 | 156 | null | other (packaging) | exp-08 | - | - | completed | accuracy 0.4867, limit 150 | inconclusive | adopt |
