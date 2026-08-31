| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 21187 | 1.58 | sft | base_model | sft_v1.jsonl (28,781) | 2e-5 / 3 | failed | - | inconclusive | abandon_line |
| exp-02 | 21390 | 1.61 | sft | base_model | sft_v1.jsonl (28,781) | 2e-5 / 3 | failed | - | inconclusive | abandon_line |
| exp-03 | 21759 | 1.69 | sft | base_model | sft_v1.jsonl (28,781) | 2e-5 / 3 | killed | - | inconclusive | abandon_line |
| exp-04 | 23390 | 1.78 | sft | base_model | sft_v1.jsonl (28,781) | 2e-5 / 3 | killed | - | inconclusive | abandon_line |
| exp-05 | 25839 | 1.90 | sft | base_model | sft_v1.jsonl (28,781) | 2e-5 / 2 | completed | none (loss 0.138, 444 steps) | inconclusive | adopt |
| exp-06 | 30591 | 4.03 | decode-config | exp-05 (runs/sft_v1/final) | - | - | completed | 0.853 @150; 0.8666 @1319 | supported | adopt |
| exp-07 | 30834 | 4.19 | decode-config | exp-05 (runs/sft_v1/checkpoint-222) | - | - | completed | 0.8467 @150 | inconclusive | reject |
| exp-08 | 33936 | 4.89 | sft | base_model | sft_v2.jsonl (30,767) | 2e-5 / 2 | completed | none (loss 0.115, 474 steps) | inconclusive | reject |
| exp-09 | 34789 | 7.15 | decode-config | exp-08 (runs/sft_v2/final) | - | - | completed | 0.873 @150 | inconclusive | reject |
| exp-10 | 34865 | 7.29 | decode-config | exp-08 (runs/sft_v2/checkpoint-237) | - | - | completed | 0.8533 @150 | inconclusive | reject |
| exp-11 | 35513 | 7.44 | decode-config | exp-08 (runs/sft_v2/final) | - | - | completed | 0.8552 @1319 | contradicted | reject |
| exp-12 | 36735 | 7.89 | decode-config | exp-05 (runs/sft_v1/final) | - | - | completed | 0.86657 @1319 | supported | adopt |
