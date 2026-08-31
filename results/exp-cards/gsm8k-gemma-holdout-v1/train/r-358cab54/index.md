# r-358cab54 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 116 | 0.18 | sft | base_model | gsm8k train (7473) | 1e-5 / 3 | completed | 0.403 @300 | inconclusive | adopt |
| exp-02 | 213 | 0.50 | decode-config | exp-01 | — | — | completed | 0.8167 @300 | supported | adopt |
| exp-03 | 270 | 0.69 | rft | base_model | gsm8k + self-RFT (28675) | 1e-5 / 2 | completed | 0.8200 @300 | inconclusive | adopt |
| exp-04 | 380 | 1.38 | grpo | exp-03 | gsm8k train prompts (7473) | 1e-6 / 1 | killed | none (killed before first save) | inconclusive | abandon_line |
| exp-05 | 437 | 1.71 | grpo | exp-03 | gsm8k train prompts (7473) | 3e-6 / 1 | killed | 0.8267 @300 (checkpoint-75) | inconclusive | adopt |
| exp-06 | 534 | 2.40 | sft | base_model | gsm8k + RFT + MetaMath 80K (108675) | 1e-5 / 2 | failed | none (duplicate save_total_limit) | inconclusive | abandon_line |
| exp-07 | 542 | 2.42 | sft | base_model | gsm8k + RFT + MetaMath 80K (108675) | 1e-5 / 2 | killed | 0.8467 @300 (checkpoint-3396, epoch 1) | supported | adopt |
| exp-08 | 647 | 3.69 | sft | base_model | gsm8k + RFT + MetaMath 130K (158675) | 1e-5 / 1 | completed | 0.8333 @300 | contradicted | reject |
| exp-09 | 782 | 5.58 | grpo | exp-07 | gsm8k train prompts (7473) | 2e-6 / 1 | failed | none (--g removed by an in-place edit) | inconclusive | abandon_line |
| exp-10 | 791 | 5.64 | grpo | exp-07 | gsm8k train prompts (7473) | 2e-6 / 1 | killed | 0.8633 @300 / 0.8537 @1319 (checkpoint-120) | supported | adopt |
| exp-11 | 869 | 6.46 | grpo | exp-10 | gsm8k train prompts (7473) | 1e-6 / 1 | killed | 0.8767 @300 / 0.8726 @1319 (checkpoint-80) | supported | adopt |
| exp-12 | 926 | 7.01 | grpo | exp-11 | gsm8k train prompts (7473) | 1e-6 / 1 | killed | 0.8867 @300 / 0.8704 @1319 (checkpoint-60) | inconclusive | adopt |
| exp-13 | 1022 | 7.87 | other (fp32→bf16 packaging) | exp-12 | — | — | completed | 0.8704 @1319; 0.8767 @300 under evaluator defaults | supported | adopt |
