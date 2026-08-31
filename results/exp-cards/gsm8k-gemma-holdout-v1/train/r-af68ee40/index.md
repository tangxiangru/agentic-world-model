| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 72 | 0.16 | sft | base_model | gsm8k-train | 1.5e-05 / 3 | completed | 0.56 @50 | inconclusive | adopt |
| exp-02 | 133 | 1.21 | sft | exp-01 | gsm8k-train + synth5k | 5e-06 / 1 | completed | 0.5667 @150 | supported | adopt |
| exp-03 | 161 | 1.83 | sft | exp-02 | gsm8k-train + synth8k | 3e-06 / 1 | killed | - | inconclusive | abandon_line |
| exp-04 | 184 | 1.87 | sft | exp-02 | gsm8k-train + synth8k | 3e-06 / 1 | killed | 0.6267 @150 | supported | adopt |
| exp-05 | 221 | 2.73 | sft | exp-04 | gsm8k-train | 2e-06 / 0.5 | completed | 0.7 @150 | contradicted | adopt |
| exp-06 | 236 | 2.94 | other | exp-04 | - | - | completed | 0.5333 @150 | contradicted | adopt |
| exp-07 | 270 | 3.02 | decode-config | exp-06 | - | - | completed | 0.6933 @150 | supported | adopt |
| exp-08 | 281 | 3.06 | decode-config | exp-07 | - | - | completed | 0.5867 @150 | contradicted | reject |
| exp-09 | 297 | 3.11 | decode-config | exp-01,exp-02,exp-04,exp-05 | - | - | completed | 0.7 @150 | inconclusive | reject |
| exp-10 | 321 | 3.26 | sft | exp-05 | gsm8k-train | 1e-06 / 0.5 | completed | 0.6933 @150 | contradicted | reject |
| exp-11 | 334 | 3.48 | sft | exp-04 | gsm8k-train | 2e-06 / 0.5 | completed | 0.7067 @150 | supported | reject |
| exp-12 | 343 | 3.69 | sft | exp-04 | gsm8k-train | 2e-06 / 0.25 | completed | 0.7267 @150 | inconclusive | adopt |
| exp-13 | 350 | 3.83 | sft | exp-04 | gsm8k-train | 2e-06 / 0.125 | completed | 0.7 @150 | inconclusive | reject |
| exp-14 | 356 | 3.93 | sft | exp-04 | gsm8k-train | 2e-06 / 0.25 | completed | 0.7 @150 | inconclusive | reject |
| exp-15 | 363 | 4.07 | other | exp-12 | - | - | completed | 0.7267 @150 | supported | adopt |
| exp-16 | 371 | 4.11 | sft | exp-04 | gsm8k-train | 2e-06 / 0.25 | completed | 0.6933 @150 | contradicted | reject |
| exp-17 | 428 | 4.30 | sft | exp-15 | gsm8k-train + orca20k | 1e-06 / 0.5 | failed | - | inconclusive | abandon_line |
| exp-18 | 460 | 4.95 | sft | exp-15 | gsm8k-train + orca20k | 1e-06 / 0.35 | completed | 0.7133 @150 | contradicted | adopt |
| exp-19 | 473 | 5.44 | sft | exp-18 | gsm8k-train | 2e-06 / 0.25 | completed | 0.7067 @150 | contradicted | reject |
| exp-20 | 487 | 5.58 | sft | exp-15 | gsm8k-train + metamath20k | 1e-06 / 0.35 | completed | 0.74 @150 | supported | adopt |
| exp-21 | 496 | 6.07 | sft | exp-20 | gsm8k-train | 2e-06 / 0.25 | completed | 0.72 @150 | contradicted | reject |
| exp-22 | 517 | 6.21 | sft | exp-15 | gsm8k-train + metamath20k | 1e-06 / 0.35 | completed | 0.72 @150 | inconclusive | reject |
| exp-23 | 531 | 6.69 | sft | exp-15 | gsm8k-train + metamath40k | 1e-06 / 0.25 | completed | 0.6933 @150 | inconclusive | reject |
| exp-24 | 545 | 7.31 | sft | exp-15 | gsm8k-train + metamath20k | 1e-06 / 0.35 | completed | 0.7133 @150 | inconclusive | reject |
| exp-25 | 558 | 7.80 | sft | exp-15 | gsm8k-train + metamath20k | 1e-06 / 0.25 | completed | 0.72 @150 | inconclusive | reject |
| exp-26 | 585 | 8.20 | other | exp-20 | - | - | completed | 0.74 @150 | inconclusive | adopt |
| exp-27 | 642 | 8.34 | decode-config | exp-26 | - | - | completed | 0.74 @150 | supported | adopt |
| exp-28 | 658 | 8.43 | sft | exp-26 | gsm8k-train | 5e-07 / 0.05 | completed | 0.72 @150 | contradicted | reject |
| exp-29 | 683 | 8.55 | merge | exp-20 | - | - | completed | 0.7133 @150 | contradicted | reject |
| exp-30 | 748 | 9.09 | sft | exp-12 | gsm8k-train + metamath20k | 1e-06 / 0.45 | completed | 0.7067 @150 | contradicted | reject |
