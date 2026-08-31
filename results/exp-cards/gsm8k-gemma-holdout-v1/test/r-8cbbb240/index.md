# r-8cbbb240 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 92 | 0.16 | sft | base_model | data/sft_gsm_only.jsonl (7273) | 2e-5 / 2 ep | completed | 0.167 @limit 30 | inconclusive | adopt |
| exp-02 | 149 | 0.65 | decode-config | exp-01 | — | — | completed | 0.300 @limit 50 | inconclusive | adopt |
| exp-03 | 181 | 0.76 | sft | exp-01 | data/sft_fewshot_v2.jsonl (29892) | 1e-5 / 1 ep | completed | 0.590 @limit 100 | inconclusive | adopt |
| exp-04 | 255 | 2.57 | sft | exp-03 | data/sft_metamath_v3.jsonl (52667) | 8e-6 / 2000 steps (0.61 ep) | completed | 0.530 @limit 100 | contradicted | reject |
| exp-05 | 296 | 4.38 | grpo | exp-03 | openai/gsm8k train (7473), zero-shot prompts | 5e-7 / 300 steps | completed | none (weights lost, disk full) | inconclusive | abandon_line |
| exp-06 | 339 | 5.71 | grpo | exp-03 | openai/gsm8k train (7473), zero-shot prompts | 5e-7 / 350 steps | killed | none (killed at step 5) | inconclusive | abandon_line |
| exp-07 | 342 | 5.72 | other (package to final_model) | exp-03 | — | — | completed | none | inconclusive | adopt |
| exp-08 | 357 | 5.75 | grpo | exp-03 | openai/gsm8k train (7473), 5-shot prompts | 5e-7 / 220 steps | completed | 0.540 @limit 100 | contradicted | reject |
| exp-09 | 423 | 6.93 | rft | exp-03 | data/sft_rft_mix.jsonl (72636 = 32690 self-generated + 14946 gold + 25000 MetaMath) | 5e-6 / 1800 steps (0.4 ep) | completed | 0.553 @limit 150 | supported | adopt |
| exp-10 | 479 | 8.48 | rft | exp-09 | data/sft_rft_clean.jsonl (47636, no MetaMath) | 3e-6 / 1200 steps (0.4 ep) | completed | 0.533 @limit 150 | contradicted | reject |
| exp-11 | 510 | 9.51 | other (package to final_model) | exp-09 | — | — | completed | 0.553 @limit 150 | inconclusive | adopt |
| exp-12 | 529 | 9.59 | sft | exp-09 | data/sft_final_push.jsonl (47636, 10-shot only) | 2e-6 / 500 steps | killed | none (killed while tokenizing) | inconclusive | abandon_line |
| exp-13 | 572 | 9.73 | merge | exp-09 (50/50 with exp-03) | — | — | completed | 0.553 @limit 150 | inconclusive | reject |
| exp-14 | 588 | 9.78 | other (package to final_model) | exp-03 | — | — | completed | 0.520 @limit 150 | inconclusive | reject |
| exp-15 | 598 | 9.81 | other (package to final_model) | exp-09 | — | — | completed | 0.553 @limit 150 | inconclusive | adopt |
