# r-ce67516f — 15 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 54 | 0.14 | sft | base_model | gsm8k train, exact prompt | 1e-5 / 2 | completed | 0.100 @ n=50 | inconclusive | adopt |
| exp-02 | 104 | 1.06 | decode-config | exp-01 | none | n/a | completed | 0.600 @ n=150 (0.460 @ n=50) | supported | adopt |
| exp-03 | 110 | 1.09 | sft | exp-02 | gsm8k train direct + MetaMathQA 100k train-derived | 1e-5 / 1 | completed | 0.593 @ n=150 | inconclusive | adopt |
| exp-04 | 153 | 2.40 | sft | exp-03 | gsm8k train, exact prompt | 5e-6 / 1 | completed | 0.607 @ n=150 | inconclusive | adopt |
| exp-05 | 196 | 2.97 | sft | exp-04 | gsm8k train, exact prompt | 2e-6 / 1 | completed | 0.567 @ n=150 | contradicted | reject |
| exp-06 | 226 | 3.46 | sft | base_model | gsm8k train direct + MetaMathQA 100k train-derived | 1e-5 / 1 | completed | 0.593 @ n=150 | inconclusive | adopt |
| exp-07 | 304 | 4.80 | sft | exp-06 | gsm8k train, exact prompt | 5e-6 / 1 | completed | 0.473 @ n=150 | contradicted | reject |
| exp-08 | 358 | 5.30 | merge | exp-04 with exp-02, alpha 0.5 | none | n/a | completed | 0.593 @ n=150 | inconclusive | reject |
| exp-09 | 367 | 5.36 | merge | exp-04 with exp-02, alpha 0.8 | none | n/a | completed | 0.567 @ n=150 | contradicted | reject |
| exp-10 | 405 | 5.47 | decode-config | exp-04 | none | n/a | completed | 0.700 @ n=150 evaluator defaults | supported | reject |
| exp-11 | 412 | 5.51 | decode-config | exp-02 | none | n/a | completed | 0.713 @ n=150 evaluator defaults | inconclusive | reject |
| exp-12 | 418 | 5.56 | decode-config | exp-03 | none | n/a | completed | 0.7468 @ n=1319 | supported | reject |
| exp-13 | 448 | 5.75 | sft | exp-02 | gsm8k train direct + MetaMathQA 200k train-derived | 1e-5 / 1 | completed | 0.7551 @ n=1319 | supported | adopt |
| exp-14 | 597 | 8.46 | other (copy to final_model) | exp-13 | none | n/a | completed | 0.7574 @ n=1319 | supported | adopt |
| exp-15 | 623 | 8.52 | sft | exp-14 | gsm8k train direct + MetaMathQA 75k train-derived | 5e-6 / 1 | failed | none (save failed, no weights) | inconclusive | abandon_line |

Notes:
- exp-14 is the submitted card: final_model holds exp-13's weights, measured 0.7574 over the full 1,319-sample test set from the final path.
- Two smoke runs ([47] LoRA, [50] full-parameter, 2 steps on 16 examples each) are recorded on exp-01 as `provenance.smoke_runs`, not as cards.
- exp-02, exp-10, exp-11 and exp-12 are config-only candidates whose creating command is filtered out of the digest; their `launch_i` points at the first event that uses the candidate (see each card's `provenance.unresolved`).
- The 150-sample evals before [394] used `--max-tokens 1024 --gpu-memory-utilization 0.5`; from [394] on they use the evaluator defaults (`max_tokens 4000`). Scores are only comparable within a group.
