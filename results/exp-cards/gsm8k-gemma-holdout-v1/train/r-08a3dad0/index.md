| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 247 | 0.19 | sft | base_model | sft_data.jsonl (101209) | 1e-5 / 3 | completed | 0.1267 @150 (eval_6094.json); sft_out 0.1133 (eval_sftout.json) | contradicted | adopt |
| exp-02 | 576 | 2.51 | decode-config | exp-01 | - | - / - | completed | 0.6133 @150 (eval_fixed.json) | supported | adopt |
| exp-03 | 637 | 2.60 | other (packaging) | exp-02 | - | - / - | completed | 0.6133 @150 (eval_final_check.json) | supported | adopt |
| exp-04 | 712 | 2.68 | sft | base_model | sft_data2.jsonl (262471) | 1e-5 / 2 | completed | 0.4733 @150 (eval_v2.json) | contradicted | reject |
| exp-05 | 806 | 5.30 | sft | base_model | sft_data2.jsonl (262471) | 1e-5 / 3 | failed | none | inconclusive | abandon_line |
| exp-06 | 841 | 5.48 | sft | base_model | sft_data2.jsonl (262471) | 1e-5 / 3 | completed | 0.0400 @150 (eval_v3.json) | contradicted | reject |
