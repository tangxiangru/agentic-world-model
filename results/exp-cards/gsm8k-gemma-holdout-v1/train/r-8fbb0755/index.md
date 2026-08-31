| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 155 | 0.32 | sft | base_model | train_data.jsonl (25000 of 167473) | 1e-5 / 1 | completed | 0.560 @50 (derisk_metrics.json) | supported | reject |
| exp-02 | 211 | 0.57 | sft | base_model | train_data.jsonl (167473) | 2e-5 / 2 (stopped at 1751/3490 steps) | killed | 0.480 @150 (v1ep1_metrics.json) | inconclusive | reject |
| exp-03 | 323 | 1.78 | sft | base_model | train_data.jsonl (167473) | 2e-5 / 1 | completed | 0.447 @150 (v2_metrics.json) | contradicted | reject |
| exp-04 | 385 | 3.07 | sft | base_model | train_data_v3mix.jsonl (120000 = 80000 brief-system + 40000 10-shot-system) | 2e-5 / 1 | completed | 0.687 @150 (v3_metrics.json) | supported | adopt |
| exp-05 | 455 | 5.23 | other (packaging) | exp-04 | none | n/a | completed | none | inconclusive | reject |
| exp-06 | 503 | 5.88 | rft | exp-04 | train_data_v4.jsonl (42258 = rft_data.jsonl 12264 + 15000 10-shot + 15000 brief) | 1e-5 / 1 | completed | 0.713 @150 (v4_metrics.json) | supported | adopt |
| exp-07 | 521 | 6.73 | other (packaging) | exp-06 | none | n/a | completed | 0.737 @300 (final_metrics.json); 0.715 @400; 0.720 @50 | inconclusive | adopt |
| exp-08 | 521 | 6.73 | grpo | exp-06 | openai/gsm8k train prompts (7473) | 1e-6 / max_steps 120 | failed | none | inconclusive | abandon_line |
| exp-09 | 532 | 6.79 | grpo | exp-06 | openai/gsm8k train prompts (7473) | 1e-6 / max_steps 120 | failed | none | inconclusive | abandon_line |
| exp-10 | 545 | 6.81 | grpo (LoRA r=32) | exp-06 | openai/gsm8k train prompts (7473) | 1e-5 / max_steps 150 | failed | none | inconclusive | abandon_line |
| exp-11 | 570 | 7.43 | rft | exp-06 | train_data_v5.jsonl (48804 = rft_data.jsonl + rft_data2.jsonl + 12000 10-shot + 12000 brief) | 1e-5 / 1 | completed | 0.707 @150 (v5_metrics.json) | contradicted | reject |
