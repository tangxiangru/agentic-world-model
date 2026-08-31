# r-ffa696e1 — experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · 10 h budget · 1x H100.
16 cards, one per launch. Accuracies are the agent's own evals; `@N` is the
`evaluate.py --limit N` they were measured under.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 139 | 0.22 | sft | base_model | gsm8k 7,473 | 1e-5 / 3 | completed | 0.047 @150 | inconclusive | adopt |
| exp-02 | 261 | 0.60 | decode-config | exp-01 | — | — | completed | 0.660 @50 | inconclusive | reject |
| exp-03 | 282 | 0.66 | sft | base_model | gsm8k x4 + metamath (429,892 rows) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-04 | 311 | 0.70 | sft | base_model | gsm8k x3 + metamath 100k | 1e-5 / 2 | failed | — | inconclusive | abandon_line |
| exp-05 | 373 | 0.89 | sft | base_model | gsm8k x3 + metamath 100k (122,419 rows) | 1e-5 / 2 | killed | 0.713 @150 | inconclusive | reject |
| exp-06 | 578 | 2.34 | rft | base_model | gsm8k x2 + rft 14,037 + metamath 60k (88,983 rows) | 1e-5 / 2 | killed | 0.753 @150 | supported | adopt |
| exp-07 | 611 | 3.03 | other (packaging) | exp-06 | — | — | completed | 0.733 @30 | inconclusive | adopt |
| exp-08 | 650 | 3.67 | rft | base_model | gsm8k x2 + rft_all 26,872 + metamath 40k (81,818 rows) | 1e-5 / 2 | killed | 0.753 @150 | contradicted | reject |
| exp-09 | 677 | 4.32 | rft | base_model | gsm8k x2 + rft_all 26,872 + metamath 40k (81,818 rows) | 2e-5 / 2 | killed | 0.700 @150 | contradicted | reject |
| exp-10 | 738 | 5.08 | grpo | exp-07 | gsm8k train prompts 7,473 | 2e-6 / 1 (beta 0.0) | completed | 0.000 @30 | contradicted | reject |
| exp-11 | 859 | 7.06 | grpo | exp-07 | gsm8k train prompts 7,473 | 1e-6 / 0.5 (beta 0.04) | completed | 0.833 @150 (ckpt-934) | supported | adopt |
| exp-12 | 896 | 7.94 | other (packaging) | exp-11 | — | — | completed | 0.847 @150; 0.810 @400; 0.788 @1319 | supported | adopt |
| exp-13 | 922 | 8.09 | grpo | exp-12 | gsm8k train prompts 7,473 | 1e-6 / 0.35 (beta 0.04) | failed | — | inconclusive | abandon_line |
| exp-14 | 935 | 8.13 | grpo | exp-11 | gsm8k train prompts 7,473 | 1e-6 / 0.3 (beta 0.04) | completed | 0.800 @150 | contradicted | reject |
| exp-15 | 999 | 8.77 | merge | exp-11 | — | — | failed | — | inconclusive | abandon_line |
| exp-16 | 1003 | 8.78 | merge | exp-11 | — | — | completed | 0.800 @150 | contradicted | reject |

**Submitted:** exp-12 — `final_model` = the KL-regularised GRPO checkpoint from
exp-11 (step 934), packaged with greedy decoding config. Verified intact at the
graded path at [1077].
