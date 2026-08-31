# r-426bd770 — reconstructed experiment cards

Base model post-trained: Qwen/Qwen3-1.7B-Base · benchmark gsm8k · 10 h budget · one 80GB GPU.
Accuracies are the agent's own evals; `@N` is the eval's `--limit` (`@1319` = full test set).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 405 | 0.17 | sft | base_model | data/sft.jsonl | 1.4e-5 / 2.0 | failed | none | inconclusive | abandon_line |
| exp-02 | 436 | 0.24 | sft | base_model | data/sft.jsonl | 1.4e-5 / 2.0 | failed | none | inconclusive | abandon_line |
| exp-03 | 525 | 0.30 | sft | base_model | data/sft.jsonl | 1.4e-5 / 2.0 | completed | 0.820 @500 (greedy); 0.676 @500 (pre-greedy config) | inconclusive | adopt |
| exp-04 | 866 | 2.35 | grpo | exp-03 | gsm8k train prompts, n=6473 | 1.5e-6 / 3.0, max_steps 400 | failed | none (died at step 40) | inconclusive | abandon_line |
| exp-05 | 918 | 5.53 | other (packaging) | exp-03 | none | none | completed | none | inconclusive | reject |
| exp-06 | 940 | 5.54 | grpo | exp-03 | gsm8k train prompts, n=6473 | 1.5e-6 / 3.0, max_steps 250 | killed | 0.858 @500 (+0.038 vs exp-03); 0.842 @1319 | supported | adopt |
| exp-07 | 1042 | 7.72 | merge | exp-03 + exp-06 | none | none | completed | 0.8522 @1319 (+0.0102 vs exp-06) | supported | adopt |
| exp-08 | 1094 | 7.80 | decode-config | exp-03 and every other candidate | none | none | completed | 0.820 @500 (+0.144 vs the same checkpoint sampled) | supported | adopt |
| exp-09 | 1139 | 7.95 | merge | exp-03 + exp-06 | none | none | failed | none | inconclusive | abandon_line |
| exp-10 | 1155 | 7.96 | merge | exp-03 + exp-06 | none | none | completed | 0.848 @1319 (soupA; -0.0042 vs exp-07) | contradicted | reject |
| exp-11 | 1192 | 8.17 | merge | exp-03 + exp-06 | none | none | completed | none (soupD eval value not in the stream) | inconclusive | reject |
| exp-12 | 1208 | 8.22 | other (packaging) | exp-07 | none | none | completed | 0.833 @150 default invocation; 0.8522 @1319 for the same weights | supported | adopt |
| exp-13 | 1242 | 8.27 | grpo | exp-06 | gsm8k train prompts, n=6473 | 1.5e-6 / 3.0, max_steps 60, seed 7 | completed | none (never evaluated on its own) | inconclusive | adopt |
| exp-14 | 1262 | 8.94 | merge | exp-03 + exp-06 + exp-13 | none | none | completed | none (soupE/soupF eval values not in the stream) | inconclusive | reject |

Submitted checkpoint: **exp-12** — final_model, the 0.5/0.5 average built in exp-07, confirmed by the
md5 check at [1281]. Smoke tests at [794], [800] and [840] are recorded on exp-04, not as cards.
