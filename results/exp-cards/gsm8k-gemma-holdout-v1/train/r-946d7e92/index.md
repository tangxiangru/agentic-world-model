# Reconstructed experiment cards - run r-946d7e92

Base model post-trained: HuggingFaceTB/SmolLM3-3B-Base. Benchmark: gsm8k, 10 h budget, one H100.
16 cards, one per launch found in the digest. Accuracies are the agent's own `evaluate.py`
runs; `@N` is the `--limit`. Runs at `--limit >= 200` were made at `--max-connections 16-48`,
which the agent later showed injects degenerate outputs for every checkpoint ([2248]); the
`@150` numbers use `evaluate.py`'s true defaults.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 417 | 0.28 | sft | base_model | sft_v1.jsonl (gsm8k train + OpenMathInstruct-2 28k) | 1e-5 / 2 ep | killed | none | inconclusive | abandon_line |
| exp-02 | 460 | 0.32 | sft | base_model | sft_v1.jsonl (gsm8k train + OpenMathInstruct-2 28k) | 1e-5 / 1 ep | completed | 0.865 @200 (also 0.863 @300, 0.834 @500, 0.847 @150 default) | supported | adopt |
| exp-03 | 1128 | 4.40 | sft | base_model | sft_v2.jsonl (rejection-sampled self-solutions + gsm8k train + OpenMathInstruct-2 3k) | 1e-5 / 2 ep | completed | 0.833 @300 | contradicted | reject |
| exp-04 | 1329 | 5.87 | grpo | exp-02 | grpo_prompts.jsonl (16014 filtered questions) | 1e-6 / 160 steps | killed | none | inconclusive | abandon_line |
| exp-05 | 1365 | 5.91 | grpo | exp-02 | grpo_prompts.jsonl (16014 filtered questions) | 1e-6 / 160 steps | failed | none | inconclusive | abandon_line |
| exp-06 | 1458 | 6.09 | grpo | exp-02 | grpo_prompts.jsonl (16014 filtered questions) | 1e-6 / 180 steps | completed | 0.880 @500 (checkpoint-120; 0.782 @500 at step 180, 0.648 @500 at step 60) | supported | adopt |
| exp-07 | 1763 | 7.12 | merge | exp-06 | none | n/a | completed | 0.878 @500 | inconclusive | reject |
| exp-08 | 1763 | 7.12 | merge | exp-02 (+ exp-06) | none | n/a | completed | 0.840 @500 | contradicted | reject |
| exp-09 | 1845 | 7.22 | other (package to final_model) | exp-06 | none | n/a | completed | none (carries exp-06's 0.880 @500) | inconclusive | adopt |
| exp-10 | 1882 | 7.23 | other (bf16 recast) | exp-06 | none | n/a | completed | 0.874 @500 (0.142 on the full 1319 at high concurrency) | inconclusive | adopt |
| exp-11 | 1929 | 7.28 | other (package to final_model) | exp-10 | none | n/a | completed | none (carries exp-10's 0.874 @500) | inconclusive | adopt |
| exp-12 | 1929 | 7.28 | grpo | exp-10 | grpo_prompts.jsonl, 35% rendered with the eval's 10-shot system message | 4e-7 / 60 steps | completed | 0.880 @500 (0.870 @500 at step 30) | inconclusive | adopt |
| exp-13 | 2009 | 8.10 | other (bf16 recast) | exp-12 | none | n/a | completed | 0.698 @500, 98/500 degenerate | contradicted | reject |
| exp-14 | 2071 | 8.17 | other (package to final_model) | exp-12 | none | n/a | completed | none (removed at [2081] for disk space) | inconclusive | abandon_line |
| exp-15 | 2089 | 8.20 | other (package to final_model) | exp-12 | none | n/a | completed | 0.853 @150 default (0.472 on the full 1319, 596 degenerate) | inconclusive | reject |
| exp-16 | 2175 | 8.46 | other (package to final_model) | exp-02 | none | n/a | completed | 0.847 @150 default (0.679 @700 at --max-connections 48) | supported | adopt |

Submitted model: **exp-16** - `final_model` is the exp-02 SFT checkpoint (`ckpt_sft1`), installed
at [2175] and verified in place at [2274], the last state of the workspace.
