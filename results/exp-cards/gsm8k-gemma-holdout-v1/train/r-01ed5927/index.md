# r-01ed5927 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 102 | 0.17 | sft | base_model | sft_gsm8k.jsonl (7,473) | 1e-5 / 3 | completed | 0.4267 @150 (eval_sft_v1.json) | contradicted | adopt |
| exp-02 | 312 | 0.59 | decode-config | exp-01 | — | — | completed | 0.8133 @150 (eval_sft_v1_full.json) | supported | adopt |
| exp-03 | 336 | 0.61 | other (package to final_model) | exp-02 | — | — | completed | none | inconclusive | adopt |
| exp-04 | 401 | 0.73 | rft | base_model | sft_combined.jsonl (27,999) | 1e-5 / 2 | completed | 0.8127 @1319 (eval_sft_v2_full.json) | inconclusive | reject |
| exp-05 | 529 | 1.38 | sft | base_model | sft_v3.jsonl (77,999) | 1e-5 / 2 | failed | none (save aborted after training) | inconclusive | iterate |
| exp-06 | 577 | 3.53 | sft | base_model | sft_v3.jsonl (77,999) | 1e-5 / 2 | completed | 0.8385 @1319 (eval_sft_v3_full.json) | supported | adopt |
| exp-07 | 663 | 5.76 | other (package to final_model) | exp-06 | — | — | completed | 0.84 @150 (eval_final_default.json) | inconclusive | adopt |
| exp-08 | 685 | 5.81 | sft | base_model | sft_v4.jsonl (99,999) | 1e-5 / 2 | completed | 0.8332 @1319 (eval_sft_v4_full.json) | contradicted | reject |
