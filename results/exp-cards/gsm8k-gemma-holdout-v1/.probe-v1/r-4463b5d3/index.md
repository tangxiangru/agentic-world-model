| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 470 | null | sft | base_model | data/sft_v1.jsonl (6,000, --limit) | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-02 | 532 | null | sft | base_model | data/sft_v1.jsonl (117,650) | 1.4e-5 / 2 | killed | none | inconclusive | abandon_line |
| exp-03 | 632 | null | sft | base_model | data/sft_v1.jsonl (117,650) | 1.4e-5 / 2 | completed | 0.695 acc @ n=200 | inconclusive | adopt |
| exp-04 | 769 | null | other (package epoch-1 ckpt) | exp-03 | none | none | completed | 0.690 acc @ n=200 | inconclusive | reject |
| exp-05 | 896 | null | other (copy to final_model) | exp-03 | none | none | completed | none | inconclusive | reject |
| exp-06 | 974 | null | grpo | exp-03 | data/rl_pool_full.jsonl (96) | 1e-6 / 1 | failed | none | inconclusive | abandon_line |
| exp-07 | 998 | null | grpo | exp-03 | data/rl_pool_full.jsonl (96) | 1e-6 / 1 | completed | none | inconclusive | abandon_line |
| exp-08 | 1040 | null | grpo | exp-03 | data/rl_pool_full.jsonl (20,000) | 1.5e-6 / 1 | failed | none | inconclusive | abandon_line |
| exp-09 | 1079 | null | grpo | exp-03 | data/rl_pool_full.jsonl (20,000) | 1.5e-6 / 1 | killed | 0.866 acc @ n=800 (0.850 @ n=200) | inconclusive | adopt |
| exp-10 | 1245 | null | other (copy to final_model) | exp-09 | none | none | completed | none | inconclusive | reject |
| exp-11 | 1247 | null | grpo | exp-09 | data/rl_pool_full.jsonl (12,000, rows 12000-24000) | 1.5e-6 / 1 | killed | 0.874 acc @ n=800 (0.890 @ n=300) | inconclusive | adopt |
| exp-12 | 1325 | null | other (copy to final_model) | exp-11 | none | none | completed | none | inconclusive | reject |
| exp-13 | 1343 | null | other (bf16 re-save + finalize) | exp-11 | none | none | completed | 0.8696 acc @ n=1319 | inconclusive | adopt |
