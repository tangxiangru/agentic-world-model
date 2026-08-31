# r-715efad9 — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 80GB
7 cards. The digest carries no timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 64 | null | sft (LoRA r=64, 4-bit) | base_model | train_data.jsonl (GSM8K + MetaMathQA GSM + MetaMathQA MATH-int; 50k of 371,323) | 2e-4 / 1 | failed | none | inconclusive | iterate |
| exp-02 | 79 | null | sft (LoRA r=64, 4-bit) | base_model | train_data.jsonl (50k of 371,323) | 2e-4 / 1 | completed | accuracy 0.52, n=50 (eval_result_v1.json) | inconclusive | reject |
| exp-03 | 116 | null | sft (LoRA r=128, 4-bit) | base_model | train_data.jsonl (80k of 371,323) | 1e-4 / 2 | completed | accuracy 0.50, n=50 (eval_result_v2.json) | contradicted | reject |
| exp-04 | 143 | null | sft (LoRA r=64, 4-bit) | base_model | train_data_v3.jsonl (GSM8K x3 + MetaMathQA GSM, tails stripped; 100k of 262,417) | 2e-4 / 1 | completed | accuracy 0.52, n=50 (eval_result_v3.json) | contradicted | adopt |
| exp-05 | 161 | null | decode-config (greedy) | exp-04 | none | — / — | completed | accuracy 0.52, n=50 (eval_result_v3_greedy.json) | contradicted | adopt |
| exp-06 | 173 | null | sft (full fine-tune, bf16) | base_model | train_data_v3.jsonl (50k of 262,417) | 5e-6 / 1 | killed | none (train loss 0.57 → 0.38, reported at [178]) | inconclusive | abandon_line |
| exp-07 | 188 | null | sft (LoRA r=64, 4-bit) | base_model | train_data_v5.jsonl (GSM8K + Orca-Math + MetaMathQA GSM; n never printed) | 2e-4 / 3 | killed | none | inconclusive | abandon_line |

Notes

- Every launch trains from `google/gemma-3-4b-pt`; only exp-05 (a decode change) has a card as its parent. Each training script writes over `/home/ben/task/final_model`, so "adopt" here means the checkpoint the agent worked from next, not a checkpoint kept aside.
- Nothing beat 52% on the agent's own 50-item evaluations: 0.52 → 0.50 → 0.52 → 0.52. No base-model number was ever measured, so no card has a base comparator.
- exp-06 and exp-07 have no result blocks in the digest (the stream ends at exp-07's launch), so their `execution: killed` is the protocol's default for a launch that stops appearing, not an observed kill.
- Which checkpoint was submitted is unresolved: see exp-07's `provenance.unresolved`.
