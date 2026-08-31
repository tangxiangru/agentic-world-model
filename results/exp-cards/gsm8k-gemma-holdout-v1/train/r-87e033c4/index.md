# r-87e033c4 - extracted experiment cards

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, 1x H100.
10 cards, launch order. `sft-data` = /home/ben/task/data/sft, 62419 records
(gsm8k train x3 + 20k MetaMathQA GSM_AnsAug + 20k GSM_Rephrased).
Smoke test at [74] (64-example, 8-step pipeline check) is recorded on exp-01, not as a card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 82 | 0.09 | sft | base_model | sft-data | 1e-5 / 1.0 | completed | 0.040 @150 (eval_v1_150.json) | inconclusive | adopt |
| exp-02 | 146 | 1.17 | decode-config | exp-01 | - | - | completed | 0.080 @150 (eval_v1_fixed.json) | contradicted | adopt |
| exp-03 | 169 | 1.35 | decode-config | exp-02 | - | - | completed | 0.060 @150 (eval_v1_fixed2.json) | contradicted | adopt |
| exp-04 | 201 | 1.54 | decode-config | exp-03 | - | - | completed | 0.013 @150 (eval_v1_greedy.json) | contradicted | reject |
| exp-05 | 224 | 1.74 | sft | base_model | sft-data | 1e-5 / 1.0 | completed | 0.233 @150 (eval_v2.json) | supported | adopt |
| exp-06 | 266 | 2.65 | decode-config | exp-05 | - | - | completed | 0.823 @1319 (eval_v2_full.json); 0.820 @150 (eval_v2_greedy.json) | supported | adopt |
| exp-07 | 294 | 2.79 | sft | base_model | sft-data | 1e-5 / 2.0 | completed | 0.807 @1319 (eval_v3_full.json); 0.813 @150 (eval_v3_150.json) | contradicted | reject |
| exp-08 | 361 | 4.61 | other (packaging) | exp-06 | - | - | completed | 0.813 @150 (eval_final_verify.json) | inconclusive | adopt |
| exp-09 | 378 | 4.62 | sft | base_model | sft-data | 5e-6 / 1.0 | completed | 0.393 @150 (eval_v4_150.json) | inconclusive | reject |
| exp-10 | 408 | 5.59 | sft | base_model | sft-data | 1.5e-5 / 1.0 | completed | 0.780 @150 (eval_v5_150.json) | inconclusive | reject |
