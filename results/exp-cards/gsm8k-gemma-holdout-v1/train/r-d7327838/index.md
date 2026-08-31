# r-d7327838 — extracted experiment cards

Base model post-trained: Qwen/Qwen3-1.7B-Base · benchmark gsm8k · 10 h budget · 1x H100 80GB.
11 launches, in launch order. `best measurement` is the agent's own eval (path relative to the run workspace).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 65 | 0.10 | sft | base_model | sft_gsm8k.jsonl (gsm8k train, 7473) | 1e-5 / 3 | completed | 0.0267 @150 (eval_v1.json) | inconclusive | adopt |
| exp-02 | 136 | 0.26 | decode-config | exp-01 | — | — / — | completed | 0.0333 @150 (eval_v1b.json) | contradicted | reject |
| exp-03 | 169 | 0.34 | sft | base_model | gsm8k train in-script (7473) + fewshot_system.txt | 1e-5 / 3 | completed | 0.3333 @150 (eval_v2.json) | supported | reject |
| exp-04 | 224 | 0.79 | sft | base_model | train_pool.jsonl (MetaMathQA GSM + gsm8k x2; 50000 used) | 1e-5 / 2 | completed | 0.5333 @150 (eval_v3.json) | supported | adopt |
| exp-05 | 288 | 2.00 | sft | base_model | train_pool_v4.jsonl (AnsAug-heavy + MATH 25k + gsm8k x3; 90000 used) | 1e-5 / 2 | completed | 0.6267 @150 (eval_v4.json); 0.598 @500 (eval_v4_500.json) | supported | adopt |
| exp-06 | 301 | 2.06 | other (packaging) | exp-04 | — | — / — | completed | none (v3 copied to final_model) | inconclusive | adopt |
| exp-07 | 324 | 4.11 | other (packaging) | exp-05 | — | — / — | completed | none (v4 copied to final_model) | inconclusive | adopt |
| exp-08 | 351 | 4.24 | rft | base_model | train_pool_v5.jsonl (rejection_pool.jsonl x3 + MetaMathQA + gsm8k; 90000 used) | 1e-5 / 2 | completed | 0.666 @500 (eval_v5_500.json); 0.6667 @150 (eval_v5.json) | supported | adopt |
| exp-09 | 390 | 6.37 | other (packaging) | exp-08 | — | — / — | completed | none (v5 copied to final_model) | inconclusive | adopt |
| exp-10 | 407 | 6.48 | rft | base_model | train_pool_v6.jsonl (rejection_pool_v2.jsonl x2 + rejection_pool.jsonl + MetaMathQA + gsm8k; 80000 used) | 1e-5 / 2 | completed | 0.688 @500 (eval_v6_500.json) | supported | adopt |
| exp-11 | 437 | 8.35 | other (packaging) | exp-10 | — | — / — | completed | 0.6667 @150 (eval_final.json) | inconclusive | adopt |

Submission: exp-11 — final_model is a copy of the exp-10 checkpoint (sft_v6), verified end-to-end at [440]-[447].
No smoke tests or dry runs appear in the digest; no scripts were written and left unrun.
