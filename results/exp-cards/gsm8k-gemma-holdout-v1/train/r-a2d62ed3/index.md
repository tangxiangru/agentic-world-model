# Extracted experiment cards — run r-a2d62ed3 (gsm8k, Qwen/Qwen3-4B-Base, 10 h, 1x H100)

Submitted model: `final_model` = the round-2 GRPO checkpoint, packaged in **exp-15** (parent exp-12).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 307 | 0.58 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | failed (OOM, no grad ckpt) | — | inconclusive | iterate |
| exp-02 | 352 | 0.62 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | failed (OOM, bs8+ckpt) | — | inconclusive | iterate |
| exp-03 | 377 | 0.65 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | failed (no space on device) | — | inconclusive | iterate |
| exp-04 | 396 | 0.67 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | killed (no outcome in stream) | — | inconclusive | iterate |
| exp-05 | 418 | 0.72 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | killed (no outcome in stream) | — | inconclusive | iterate |
| exp-06 | 433 | 0.73 | sft | base_model | sft_data_v1 (90,910) | 1e-5 / 2 ep | completed (640 steps, 1.77 h) | 0.8733 @150 — results_sft_v1_ep2.json (ep1 0.86) | supported | adopt |
| exp-07 | 642 | 2.83 | rft | exp-06 | sft_data_v2 (57,236) | 6e-6 / 1.5 ep | failed (invalid gen config at save, step 231/347) | — | inconclusive | iterate |
| exp-08 | 697 | 3.52 | rft | exp-06 | sft_data_v2 (57,236) | 6e-6 / 1 ep | completed (231 steps, 0.64 h) | 0.8933 @150 — results_sft_v2.json | supported | adopt |
| exp-09 | 706 | 3.53 | other (package → final_model) | exp-06 | — | — | completed | — (never evaluated as final_model) | inconclusive | adopt |
| exp-10 | 735 | 4.22 | grpo | exp-08 | gsm8k train prompts (7,473) | 1e-6 / 120 steps | failed (OOM on fp32 logits, bs64) | — | inconclusive | iterate |
| exp-11 | 759 | 4.26 | grpo | exp-08 | gsm8k train prompts (7,473) | 1e-6 / 120 steps | completed (0.90 h) | 0.90 @150 — results_grpo_final.json (ckpt-80 0.8867) | supported | adopt |
| exp-12 | 827 | 5.25 | grpo | exp-11 | gsm8k train prompts (7,473) | 2e-6 / 200 steps | completed (1.51 h) | 0.92 @150 — results_grpo2_final.json (0.9175 @400) | supported | adopt |
| exp-13 | 838 | 5.25 | merge | exp-08 (+ exp-11, alpha 0.5) | — | — | completed | — (never evaluated) | inconclusive | abandon_line |
| exp-14 | 904 | 6.83 | grpo | exp-12 | gsm8k train prompts (7,473) | 2e-6 / 150 steps | failed (disk full at save; weights intact) | 0.92 @150 — results_grpo3_150.json (0.915 @400) | inconclusive | reject |
| exp-15 | 982 | 8.14 | other (package → final_model) | exp-12 | — | — | completed | 0.92 @150 default args — results_final_default2.json (first run 0.9133) | supported | adopt |
