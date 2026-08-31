| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 16794 | 0.50 | sft | base_model | sft_v1_f.jsonl (86,753) | 1e-5 / 1 | failed | none (OOM on step 1) | inconclusive | abandon_line |
| exp-02 | 23266 | 0.98 | sft | base_model | sft_v2.jsonl | 1e-5 / 2 | completed | 0.7407 @ n=1319 (ep2); 0.773 @ n=150 | supported | adopt |
| exp-03 | 27964 | 3.13 | rft | exp-02 | rft_r1.jsonl + sft_v3.jsonl | 5e-6 / 1 | completed | 0.7688 @ n=1319 (+0.0281) | supported | adopt |
| exp-04 | 36125 | 5.80 | dpo | exp-03 | labeled_r2.jsonl + dpo_r2.jsonl (6,667) | 5e-7 / 1 | completed | 0.7832 @ n=1319 (+0.0144) | supported | adopt |
| exp-05 | 37006 | 7.06 | dpo | exp-04 | labeled_r3.jsonl + dpo_r3.jsonl (6,157) | 4e-7 / 1 | failed | none (weights lost at save) | inconclusive | abandon_line |
| exp-06 | 37131 | 7.06 | other | exp-04 | none | none | completed | none (packaging: dpo_v1 -> final_model) | inconclusive | reject |
| exp-07 | 38094 | 7.83 | dpo | exp-04 | dpo_r3.jsonl (6,157) | 4e-7 / 1 | completed | 0.7953 @ n=1319 (+0.0121) | supported | adopt |
| exp-08 | 39167 | 8.62 | other | exp-07 | none | none | failed | none (promotion aborted, exit 144) | inconclusive | abandon_line |
| exp-09 | 39972 | 8.65 | other | exp-07 | none | none | completed | 0.8175 @ n=400 (official path) | inconclusive | adopt |
