| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 77 | null | sft (LoRA r=64) | base_model | train_data.jsonl (27,473) | 2e-4 / 2 | failed | - | inconclusive | iterate |
| exp-02 | 89 | null | sft (LoRA r=64) | base_model | train_data.jsonl (27,473) | 2e-4 / 2 | completed | 0.080 @ n=50 (eval_iter1.json) | inconclusive | reject |
| exp-03 | 178 | null | sft (LoRA r=128, assistant_only_loss, im_end eos) | base_model | train_data.jsonl (27,473) | 1e-4 / 2 | completed | 0.647 @ n=150 (eval_v2_150.json) | supported | iterate |
| exp-04 | 221 | null | sft (full fine-tune) | base_model | train_data.jsonl (27,473) | 2e-5 / 3 | completed | 0.573 @ n=150 (eval_v3.json) | contradicted | reject |
| exp-05 | 264 | null | sft (LoRA r=128) | base_model | train_data_v2.jsonl (59,892; gsm8k x4 + 30k MetaMathQA) | 1e-4 / 2 | killed | - | inconclusive | abandon_line |
| exp-06 | 277 | null | sft (LoRA r=128) | base_model | train_data.jsonl (27,473) | 1e-4 / 3 | killed | - | inconclusive | abandon_line |
| exp-07 | 294 | null | sft (LoRA r=128) | base_model | train_data_v3.jsonl (27,473; <<calc>> markers kept) | 1e-4 / 3 | completed | 0.627 @ n=150 (eval_v6.json) | contradicted | reject |
| exp-08 | 336 | null | merge (epoch-2 adapter) | exp-07 | - | - / - | completed | 0.553 @ n=150 (eval_v6_ep2.json) | inconclusive | reject |
| exp-09 | 351 | null | sft (LoRA r=128, exp-03 replication) | base_model | train_data.jsonl (27,473) | 1e-4 / 2 | completed | 0.567 @ n=150 (eval_v7.json) | contradicted | adopt |
| exp-10 | 371 | null | sft (LoRA r=128, few-shot contexts) | base_model | train_data_v4.jsonl (22,473; 5,193 with few-shot system) | 1e-4 / 2 | killed | - | inconclusive | abandon_line |
| exp-11 | 384 | null | sft (LoRA r=128, dropout 0.0) | base_model | train_data.jsonl (27,473) | 1e-4 / 2 | killed | - | inconclusive | abandon_line |
