# Experiment cards — gsm8k / Qwen3-4B-Base / 10 h / 1x H100

Reconstructed from the run's event stream. 11 launches carry a card; five pipeline
smoke tests do not (they are listed on the card of the launch that followed them).
No baseline eval of the base model was ever run, so no card has a base_model
comparator.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 91 | 0.20 | sft | base_model | sft_combined.jsonl (262,419; MetaMath-GSM 240k + gsm8k train x3) | 2e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-02 | 145 | 0.38 | sft | base_model | sft_combined_100k.jsonl (129,892; MetaMath-GSM 100k + gsm8k train x4) | 2e-5 / 2 | completed | 0.7667 @ n=150 (eval_sft_v2_l150.json) | inconclusive | adopt |
| exp-03 | 301 | 3.20 | other (package) | exp-02 | — | — / — | completed | — | inconclusive | reject |
| exp-04 | 301 | 3.20 | grpo | exp-02 | openai/gsm8k train (7,473), answer-match reward | 5e-7 / 1 (400 steps) | completed | 0.78 @ n=150 (eval_grpo_v1_l150.json) | inconclusive | adopt |
| exp-05 | 346 | 5.56 | sft | exp-04 | sft_metamath_rest.jsonl (169,892; MetaMath-GSM 100k–240k + gsm8k train x4) | 8e-6 / 1 | completed | 0.78 @ n=150 (eval_sft_v3_l150.json) | inconclusive | reject |
| exp-06 | 405 | 7.42 | other (package) | exp-04 | — | — / — | completed | — | inconclusive | reject |
| exp-07 | 421 | 9.18 | rft | exp-04 | sft_rft_mix.jsonl (27,677; 12,731 self-sampled correct + gsm8k train x2) | 5e-6 / 2 | failed (save; step-600 salvaged) | 0.8067 @ n=150 (eval_sft_rft_l150.json) | inconclusive | adopt |
| exp-08 | 453 | 9.72 | other (package) | exp-07 | — | — / — | completed | 0.7467 @ n=150 (eval_final_l150.json) | inconclusive | reject |
| exp-09 | 475 | 9.79 | decode-config | exp-08 | — | — / — | completed | — (both 50-sample rechecks unreported) | inconclusive | reject |
| exp-10 | 485 | 9.84 | decode-config | exp-09 | — | — / — | completed | 0.78 @ n=150 (eval_final_l150_fixed.json) | inconclusive | reject |
| exp-11 | 508 | 9.93 | other (package) | exp-07 | — | — / — | completed | — (confirming eval never finished) | inconclusive | adopt |

Submitted artifact: **exp-11** — final_model as a wholesale copy of the exp-07 RFT
checkpoint. Its weights last measured 0.8067 at n=150 under the checkpoint path
(exp-07); no eval of final_model completed after that copy.
