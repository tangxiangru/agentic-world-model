# Reconstructed experiment cards — r-7842f260

Base model post-trained: `Qwen/Qwen3-1.7B-Base` · benchmark: gsm8k · budget: 10 h, 1x H100.
10 launches, in launch order. Adopted / submitted: **exp-09** (`final_model`, a copy of the exp-03 checkpoint).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 12086 | 0.19 | sft | base_model | gsm8k_sft_v1.jsonl (7,473 gold) | 1e-5 / 3 | failed | — (OOM before training) | inconclusive | iterate |
| exp-02 | 12539 | 0.22 | sft | base_model | gsm8k_sft_v1.jsonl (7,473 gold) | 1e-5 / 3 | completed | 0.660 @ limit 150 | inconclusive | adopt |
| exp-03 | 19507 | 2.25 | distill | base_model | gsm8k_sft_v2.jsonl (19,703 = teacher + self-RS + gold backfill) | 1e-5 / 2 | completed | 0.7847 @ full 1319 (0.843 @ 300) | inconclusive | adopt |
| exp-04 | 22143 | 4.20 | merge | exp-02 (+ exp-03) | — | — | killed | — (superseded by exp-05) | inconclusive | iterate |
| exp-05 | 22789 | 4.29 | merge | exp-02 (+ exp-03) | — | — | completed | — (never evaluated) | inconclusive | abandon_line |
| exp-06 | 23003 | 4.40 | distill | base_model | gsm8k_sft_v3.jsonl (1 teacher + 2 round-2 self per question) | 1e-5 / 2 | completed | 0.793 @ limit 300 | contradicted | reject |
| exp-07 | 24174 | 6.52 | merge | exp-03 (+ exp-06) | — | — | completed | 0.833 @ limit 300 | inconclusive | reject |
| exp-08 | 24999 | 6.69 | distill | base_model | gsm8k_sft_v4.jsonl (2 teacher + 1 round-2 self per question) | 1e-5 / 2 | completed | 0.784 @ full 1319 | inconclusive | reject |
| exp-09 | 25224 | 6.75 | other (packaging) | exp-03 | — | — | completed | 0.820 @ default limit 150 | inconclusive | adopt |
| exp-10 | 26690 | 8.99 | merge | exp-03 (+ exp-08) | — | — | completed | — (both evals crashed: CUDA illegal memory access) | inconclusive | abandon_line |

Comparators are recorded per card. Note that the base-model reading (0.090) was taken at `--limit 100`
and exp-02 at `--limit 150`, so no candidate has a same-protocol delta against the base model; only
exp-06/exp-07 (limit 300 vs exp-03) and exp-08/exp-10 (full test vs exp-03) share a protocol with
their comparator.
