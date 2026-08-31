# Reconstructed experiment cards - run r-46a821e3

Base model: Qwen/Qwen3-1.7B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
5 launches carry a card. Submitted directory: `final_model` = exp-05, a copy of exp-03's merged
checkpoint (`runs/run2_merged`). No trained checkpoint was ever measured above the base model's
0.12, and the sanity eval of `final_model` failed to start vLLM [186] and was never retried.
The digest records no turn timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [35] | null | sft | base_model | gsm8k-train, chat template, minus 512 held out | 2e-05 / 4ep | completed | 0.010 (n=100) | contradicted | reject |
| exp-02 | [121] | null | sft | base_model | gsm8k-train, plain Reasoning/ANSWER, minus 512 held out | 2e-04 / 3ep (killed @ ckpt-210) | killed | - | inconclusive | adopt |
| exp-03 | [142] | null | merge | exp-02 | - | - | completed | 0.100 (n=50) | inconclusive | adopt |
| exp-04 | [153] | null | sft | base_model | gsm8k-train, 2-shot in-context prompts, minus 512 held out | 1e-04 / 2ep (killed @ step 70) | killed | - | inconclusive | abandon_line |
| exp-05 | [182] | null | other | exp-03 | - | - | completed | - | inconclusive | adopt |

Reference point, not a card: the base model measured 0.120 at `--limit 100 --max-tokens 1024`
(`baseline_metrics.json`, [14]/[22]).

Not a card, no launch block in the digest: the repackaging of `runs/run1` with the base Qwen
tokenizer into `runs/run1_fixed` between [99] and [104]. Its score (0.02 at n=50,
`run1_fixed_metrics_50.json`) is recorded as a second measurement on exp-01.
