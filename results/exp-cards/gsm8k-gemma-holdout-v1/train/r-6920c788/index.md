# r-6920c788 — reconstructed experiment cards (gsm8k, SmolLM3-3B-Base, 10 h, 1x H100)

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 144 | 0.22 | sft | base_model | gsm8k_sft.jsonl (7394) | 2e-5 / 3 | completed | 0.54 @100 | supported | adopt |
| exp-02 | 248 | 0.48 | sft | base_model | combined_sft.jsonl (87394: gsm8k + 80K MetaMathQA) | 2e-5 / 3 | completed | 0.47 @100 (ep2) | contradicted | reject |
| exp-03 | 636 | 3.31 | other (package) | exp-01 | — | — | completed | none | inconclusive | adopt |
| exp-04 | 679 | 3.44 | rft | base_model | star_sft.jsonl (gsm8k + STaR r1, cap 3) | 2e-5 / 3 | completed | 0.64 @100 (ep3) | supported | adopt |
| exp-05 | 814 | 4.16 | other (package) | exp-04 | — | — | completed | none | inconclusive | adopt |
| exp-06 | 858 | 4.28 | rft | base_model | star_sft_v2.jsonl (gsm8k + STaR r1-r2, cap 5) | 2e-5 / 4 | completed | 0.74 @100 (ep4); 0.6425 @400 | supported | adopt |
| exp-07 | 964 | 5.66 | other (package) | exp-06 | — | — | completed | 0.647 @150 | inconclusive | adopt |
| exp-08 | 1006 | 5.83 | rft | base_model | star_sft_v2.jsonl (43429: gsm8k + STaR r1-r3, cap 5) | 2e-5 / 5 | completed | 0.6933 @150 (ep5); 0.6232 full | inconclusive | adopt |
| exp-09 | 1100 | 7.57 | other (package) | exp-08 | — | — | completed | 0.6275 @400 | contradicted | adopt |
| exp-10 | 1114 | 7.65 | other (package) | exp-06 | — | — | completed | 0.6171 full (1319) | contradicted | adopt |
| exp-11 | 1156 | 7.85 | other (package) | exp-08 | — | — | completed | 0.6533 @150 | inconclusive | adopt |
| exp-12 | 1169 | 7.90 | decode-config | exp-11 | — | — | completed | 0.6884 full (1319) greedy | supported | adopt |
| exp-13 | 1192 | 8.02 | decode-config | exp-06 | — | — | completed | 0.696 full (1319) greedy | contradicted | adopt |
| exp-14 | 1207 | 8.11 | other (package) | exp-06 | — | — | completed | 0.7036 full (1319), 0.7067 @150 | inconclusive | adopt |
| exp-15 | 1278 | 8.38 | rft | base_model | star_sft_v2.jsonl (36631: gsm8k + STaR r1-r4, cap 4) | 2e-5 / 3 | completed | 0.6914 full (1319) greedy | inconclusive | reject |

Notes: exp-14 is the state `final_model` was left in at the end of the run (exp-06's epoch-4 checkpoint packaged with the exp-12 greedy decode config). Every training launch loads the base model — `scripts/train_sft.py` hardcodes it — so the chain runs through the data, not the weights. Measurement subsets differ (@100/@150/@400/full) and the eval sampled non-deterministically until exp-12; deltas across different `--limit` values are not comparable.
