# r-2f4530d4 — experiment cards (gsm8k, Qwen/Qwen3-4B-Base, 10 h, 1x H100)

The digest carries no timestamps, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 61 | — | sft (LoRA r=64) | base_model | gsm8k train + MetaMathQA GSM (50k of 247,470) | 2e-4 / 1 | failed | — | inconclusive | iterate |
| exp-02 | 71 | — | sft (LoRA r=64) | base_model | gsm8k train + MetaMathQA GSM (50k of 247,470) | 2e-4 / 1 | completed | — (loss 0.24) | inconclusive | adopt |
| exp-03 | 84 | — | merge | exp-02 | — | — | completed | 0.060 @50 | inconclusive | adopt |
| exp-04 | 104 | — | decode-config | exp-03 | — | — | completed | 0.080 @50 | inconclusive | adopt |
| exp-05 | 121 | — | decode-config | exp-04 | — | — | completed | 0.040 @50 | inconclusive | abandon_line |
| exp-06 | 139 | — | sft (full) | base_model | gsm8k train 7,473 + MetaMathQA GSM 40,000 | 2e-5 / 2 | completed | 0.380 @50 | inconclusive | reject |
| exp-07 | 159 | — | sft (full) | base_model | gsm8k train 7,473 + MetaMathQA GSM 50,000 | 2e-5 / 2 | completed | 0.747 @150 (0.740 @50) | supported | adopt |
| exp-08 | 192 | — | sft (full) | base_model | gsm8k train 7,473 x3 + MetaMathQA GSM 239,997 + ORCA-Math 30,000 | 2e-5 / 2 (killed at 5135/9770) | killed | 0.760 @150 (checkpoint-4000) | inconclusive | adopt |
| exp-09 | 238 | — | other (packaging) | exp-08 | — | — | completed | — | inconclusive | adopt |
| exp-10 | 245 | — | decode-config | exp-09 | — | — | completed | 0.733 @150 | inconclusive | reject |
| exp-11 | 255 | — | decode-config | exp-10 | — | — | completed | 0.747 @150 | inconclusive | adopt |

Submission: `final_model` after exp-11 — the exp-08 checkpoint-4000 weights with the
checkpoint's original generation config. Base-model reference: 0.400 @20 (event [31]).
