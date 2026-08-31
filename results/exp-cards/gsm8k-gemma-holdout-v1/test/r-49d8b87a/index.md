# r-49d8b87a — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 127 | 0.29 | sft | base_model | data/gsm8k_train.jsonl | 2e-4 / 3 | completed | 0.4667 @150 (results/run1_gsm8k.json) | inconclusive | adopt |
| exp-02 | 256 | 0.67 | sft | base_model | data/gsm8k_mm.jsonl (39473) | 2e-4 / 2 | completed | 0.5067 @150 (results/run2_mm.json) | inconclusive | adopt |
| exp-03 | 412 | 2.41 | decode-config | exp-02 | none | n/a | completed | 0.6467 @150 (results/run2_greedy.json), 0.646 @500 | supported | adopt |
| exp-04 | 428 | 2.47 | sft (full fine-tune) | base_model | data/gsm8k_mm.jsonl (39473) | 1e-5 / 2 | completed | 0.4533 @150 (results/run3_full_greedy.json) | contradicted | reject |
| exp-05 | 440 | 2.51 | other (packaging to final_model) | exp-03 | none | n/a | completed | 0.64 @150 (results/final_confirm.json) | supported | adopt |
| exp-06 | 482 | 3.90 | sft | base_model | data/gsm8k_mm.jsonl (39473) | 2e-4 / 3 | completed | 0.64 @150 (results/run4_r64_greedy.json), 0.608 @500 | contradicted | reject |
| exp-07 | 707 | 6.94 | rft | base_model | data/rft_combined.jsonl (rft_clean 9890 + 22000 MetaMath) | 2e-4 / 2 | completed | 0.644 @500 (results/rft_n500.json), 0.6267 @150 | inconclusive | reject |

Submitted card: **exp-05** — `/home/ben/task/final_model`, the exp-02 weights carrying the exp-03 greedy `generation_config.json`.
