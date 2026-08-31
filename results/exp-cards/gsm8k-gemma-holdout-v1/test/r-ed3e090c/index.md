# r-ed3e090c — reconstructed experiment cards

Base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100.
13 cards. Every measurement is the agent's own `evaluate.py --limit 150` run.
`official_accuracy` is not written on any card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 87 | 0.12 | sft | base_model | train.jsonl 52,419 (gsm8k x3 + MetaMathQA GSM 30k) | 2e-4 / 1 | completed | — (adapter not scored) | inconclusive | adopt |
| exp-02 | 168 | 1.20 | merge | exp-01 | — | — | completed | 0.367 (n=150, eval_v1.json) | inconclusive | adopt |
| exp-03 | 208 | 1.29 | decode-config | exp-02 | — | — | completed | 0.327 (n=150, eval_v1_greedy.json) | contradicted | reject |
| exp-04 | 233 | 1.36 | sft | base_model | train_v2.jsonl 74,892 (gsm8k x4 + MetaMathQA GSM 25k + OpenMathInstruct-2 gsm 20k) | 1.5e-4 / 1 | completed | — (adapter not scored) | inconclusive | adopt |
| exp-05 | 249 | 2.83 | merge | exp-04 | — | — | completed | 0.327 (n=150, eval_v2.json) | contradicted | adopt |
| exp-06 | 254 | 2.88 | decode-config | exp-05 | — | — | completed | 0.293 (n=150, eval_v2_sample.json) | contradicted | reject |
| exp-07 | 268 | 2.94 | decode-config | exp-06 | — | — | completed | 0.320 (n=150, eval_v2_eotfix.json) | inconclusive | reject |
| exp-08 | 276 | 2.99 | decode-config | exp-07 | — | — | completed | 0.267 (n=150, eval_v2_eot2.json) | contradicted | reject |
| exp-09 | 281 | 3.04 | decode-config | exp-08 | — | — | completed | 0.240 (n=150, eval_v2_revert.json) | contradicted | reject |
| exp-10 | 300 | 3.10 | merge | exp-01 | — | — | completed | 0.373 (n=150, eval_v1_restored.json) | inconclusive | adopt |
| exp-11 | 316 | 3.16 | sft | base_model | train_v3.jsonl 77,365 (gsm8k x5 + MetaMathQA GSM 40k) | 5e-6 / 1 (full FT) | completed | — (weights not scored) | inconclusive | adopt |
| exp-12 | 337 | 4.80 | other (package) | exp-11 | — | — | completed | 0.040 (n=150, eval_v3.json) | contradicted | reject |
| exp-13 | 355 | 4.86 | merge | exp-01 | — | — | completed | 0.333 (n=150, no file; agent's report at [370]) | inconclusive | adopt |

Submitted: **exp-13** — the v1 LoRA (r=64) merge left in `final_model`, best measured 0.373 as exp-10 and 0.333 on re-verification.
