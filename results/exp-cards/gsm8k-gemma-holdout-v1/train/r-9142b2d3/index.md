| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 52 | null | sft (LoRA r=128) | base_model | gsm8k + orca-math (27,295 total) | 2e-4 / 3 | failed | none | inconclusive | iterate |
| exp-02 | 59 | null | sft (LoRA r=128) | base_model | gsm8k + orca-math | 2e-4 / 3 | killed | none | inconclusive | iterate |
| exp-03 | 84 | null | sft (LoRA r=128) | base_model | gsm8k x2 + orca-math 19,947 (34,893 total) | 2e-4 / 2 | completed | 0.133 @ n=30 | inconclusive | reject |
| exp-04 | 119 | null | sft (full FT) | base_model | gsm8k 7,473 | 2e-5 / 5 | completed | 0.7267 @ n=150 (0.700 @ n=30 vs exp-03 0.133) | supported | adopt |
| exp-05 | 150 | null | decode-config | exp-04 | none | null / null | completed | 0.620 @ n=50 | inconclusive | reject |
| exp-06 | 156 | null | decode-config | exp-04 | none | null / null | completed | 0.760 @ n=50 (vs exp-05 0.620) | inconclusive | adopt |
| exp-07 | 168 | null | other (checkpoint selection) | exp-04 | none | null / null | completed | 0.680 @ n=50 (vs exp-06 0.760) | inconclusive | reject |
| exp-08 | 182 | null | sft (full FT) | base_model | gsm8k x3 + orca-math 15k + metamath MATH-subset 15k | 2e-5 / 3 | completed | 0.7533 @ n=150 (vs exp-04 0.7267) | inconclusive | adopt |
| exp-09 | 200 | null | sft (full FT) | exp-08 | gsm8k x3 | 1e-5 / 3 | killed | none | inconclusive | abandon_line |
