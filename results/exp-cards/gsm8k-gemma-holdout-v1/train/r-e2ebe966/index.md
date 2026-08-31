# r-e2ebe966 — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h, one H100.
Digest has no per-event timestamps, so every `elapsed_h` is null.
11 launches carded; no smoke runs. `exp-11` is the adopted/submitted card
(final_model = a copy of the no-op checkpoint trained in `exp-10`).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 67 | null | sft | base_model | gsm8k train (full) | 1e-4 / 3 ep | completed | 0.000 @ n=20 (exp1_limit20.json) | contradicted | reject |
| exp-02 | 211 | null | sft | base_model | gsm8k train (full) | 1e-4 / 2 ep | completed | 0.150 @ n=20; 0.200 @ n=60 | contradicted | reject |
| exp-03 | 239 | null | sft | base_model | gsm8k train (full) | 2e-5 / 1 ep | completed | 0.100 @ n=20 | contradicted | reject |
| exp-04 | 264 | null | sft | base_model | gsm8k train (3000, 5-shot ctx) | 2e-5 / 1 ep | completed | 0.100 @ n=20 | contradicted | reject |
| exp-05 | 292 | null | sft | base_model | gsm8k train (3000, raw-prefix fmt) | 2e-5 / 1 ep | completed | 0.100 @ n=20 | contradicted | reject |
| exp-06 | 307 | null | sft | base_model | gsm8k train (full, 10 steps) | 1e-5 / 10 steps | completed | 0.300 @ n=20; 0.300 @ n=60 | contradicted | reject |
| exp-07 | 371 | null | sft | base_model | gsm8k train (full, 40 steps) | 8e-6 / 40 steps | completed | 0.400 @ n=30; 0.383 @ n=60 | contradicted | reject |
| exp-08 | 439 | null | sft | base_model | gsm8k train (full, 10-shot ctx, 20 steps) | 5e-6 / 20 steps | completed | 0.400 @ n=30; 0.383 @ n=60 | contradicted | reject |
| exp-09 | 463 | null | sft | base_model | gsm8k train (answer-only, 80 steps) | 5e-6 / 80 steps | completed | 0.400 @ n=30 | inconclusive | reject |
| exp-10 | 479 | null | sft | base_model | gsm8k train (7473 tokenized, 1 no-op step) | 0 / 1 step | completed | 0.5667 @ n=60 (exp10_limit60_mt2048.json) | supported | adopt |
| exp-11 | 490 | null | other (packaging) | exp-10 | — | — | completed | 0.300 @ n=20 (final_model_limit20.json) | inconclusive | adopt |

Comparators (base model, agent's own evals): 0.5667 @ --limit 60 --max-tokens 2048
(`baseline_limit60.json`), 0.300 @ --limit 20 --max-tokens 768
(`experiments/baseline_limit20.json`), 0.400 @ --limit 30 --max-tokens 1536
(`experiments/baseline_limit30_mt1536.json`).
