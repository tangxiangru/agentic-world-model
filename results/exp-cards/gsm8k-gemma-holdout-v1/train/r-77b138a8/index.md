| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 82 | 0.13 | sft | base_model | gsm8k-train | 2e-4 / 3 ep (killed in warmup) | killed | - | inconclusive | abandon_line |
| exp-02 | 102 | 0.18 | sft | base_model | gsm8k-train | 2e-4 / 3 ep (killed at 600 steps) | killed | - | inconclusive | adopt |
| exp-03 | 157 | 0.74 | merge | exp-02 | - | - | completed | 0.0067 (n=150) | inconclusive | reject |
| exp-04 | 185 | 0.82 | decode-config | exp-03 | - | - | completed | 0.0267 (n=150) | contradicted | reject |
| exp-05 | 190 | 0.88 | decode-config | exp-03 | - | - | completed | 0.0000 (n=20) | inconclusive | reject |
| exp-06 | 200 | 0.92 | sft | base_model | gsm8k-train | 2e-4 / 300 steps | failed | - | inconclusive | abandon_line |
| exp-07 | 206 | 0.93 | sft | base_model | gsm8k-train | 2e-4 / 300 steps | completed | 0.8133 (n=150) | supported | reject |
| exp-08 | 225 | 1.28 | sft | base_model | gsm8k-train | 2e-4 / 600 steps | completed | 0.6867 (n=150) | supported | adopt |
| exp-09 | 280 | 1.91 | sft | base_model | gsm8k-train (exact-system only) | 2e-4 / 900 steps | killed | - | inconclusive | abandon_line |
| exp-10 | 328 | 2.39 | sft | base_model | metamath_gsm_80k.jsonl | 2e-4 / 2000 steps | killed | - | inconclusive | adopt |
| exp-11 | 349 | 2.93 | merge | exp-10 | - | - | completed | 0.7000 (n=150) | contradicted | adopt |
| exp-12 | 357 | 2.97 | sft | exp-11 | gsm8k-train (exact-system only) | 5e-5 / 600 steps | completed | 0.6333 (n=150) | contradicted | reject |
| exp-13 | 372 | 3.9 | merge | exp-12 | - | - | completed | 0.7933 (n=150) | contradicted | reject |
| exp-14 | 377 | 3.93 | sft | exp-11 | gsm8k-train (clean-rationale, no-exact-system) | 2e-5 / 300 steps | completed | 0.8267 (n=150) | contradicted | reject |
| exp-15 | 434 | 4.45 | sft | exp-11 | gsm8k-train (answer-only loss) | 5e-5 / 300 steps | completed | - | inconclusive | abandon_line |
| exp-16 | 483 | 4.92 | merge | exp-15 | - | - | completed | 0.3467 (n=150) | contradicted | reject |
| exp-17 | 484 | 4.92 | merge | exp-15 | - | - | completed | - | inconclusive | abandon_line |
| exp-18 | 503 | 4.97 | sft | exp-11 | gsm8k-train (exact-system only) | 2e-5 / 200 steps | completed | 0.8200 (n=150) | contradicted | reject |
| exp-19 | 513 | 5.28 | merge | exp-18 | - | - | completed | 0.6467 (n=150) | contradicted | reject |
| exp-20 | 563 | 5.52 | sft | exp-08 | gsm8k-train | 5e-5 / 300 steps | killed | - | inconclusive | abandon_line |
| exp-21 | 611 | 5.75 | sft | base_model | synthgsm_50k.jsonl | 2e-4 / 1500 steps | failed | - | inconclusive | abandon_line |
| exp-22 | 623 | 5.76 | sft | base_model | synthgsm_50k.jsonl | 2e-4 / 1500 steps | killed | - | inconclusive | abandon_line |
| exp-23 | 643 | 6.18 | sft | base_model | gsm8k-train + synthgsm_10k.jsonl | 2e-4 / 1200 steps | killed | - | inconclusive | abandon_line |
| exp-24 | 660 | 6.5 | sft | exp-11 | gsm8k-train (eos-only loss) | 5e-5 / 100 steps | completed | 0.3600 (n=150) | contradicted | reject |
| exp-25 | 674 | 6.7 | sft | base_model | gsm8k-train | 2e-4 / 1200 steps | killed | - | inconclusive | abandon_line |
| exp-26 | 689 | 7.29 | other | exp-08 | - | - | completed | 0.6400 (n=150) | contradicted | reject |
| exp-27 | 696 | 7.33 | other | exp-08 | - | - | completed | 0.6800 (n=150) | inconclusive | adopt |
| exp-28 | 729 | 7.43 | decode-config | exp-27 | - | - | completed | 0.8333 (n=150) | supported | adopt |
