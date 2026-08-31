# r-2c6de1f4 — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on 1x H100 80GB.
The digest carries no timestamps, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 65 | null | sft (LoRA r=64) | base_model | MetaMathQA 100k (\boxed{} targets) | 2e-4 / 2 | completed | 0.380 @ n=50 | inconclusive | reject |
| exp-02 | 85 | null | sft (LoRA r=64) | base_model | gsm8k-train 7473 x5 + MetaMathQA 100k, ANSWER: format | 2e-4 / 3 | completed | 0.420 @ n=50 (+0.040 vs exp-01) | supported | reject |
| exp-03 | 107 | null | sft (LoRA r=64) | base_model | gsm8k-train 7473 x8 + MetaMathQA GSM_* + 30k MATH_*, think tags | 2e-4 / 3 | completed | 0.220 @ n=50 (-0.200 vs exp-02) | contradicted | reject |
| exp-04 | 126 | null | sft (full fine-tune) | base_model | gsm8k-train 7473 x10 + MetaMathQA 260k, no packing | 2e-5 / 3 | completed | 0.480 @ n=50 (+0.060 vs exp-02); 0.500 @ n=20 | supported | adopt |
| exp-05 | 146 | null | sft (LoRA r=128) | base_model | gsm8k-train x5 + MetaMathQA GSM_* + 15k MATH_*, few-shot system messages | 1e-4 / 2 | killed (~82%, out of budget) | none | inconclusive | abandon_line |

Not cards: [46] and [56] (trainer-API crashes before any step — recorded as
`provenance.smoke_runs` on exp-01) and [151] (`cp -r ./final_model
./final_model_v4_backup`, a backup of the exp-04 checkpoint that produced no new
candidate — recorded in exp-04's provenance).
