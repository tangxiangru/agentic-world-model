# r-78f13a5c — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on one H100.
Ten launches carded, in launch order. `official_accuracy` is not written by the
extractor and is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 94 | 0.27 | sft | base_model | combined_math_train.jsonl (247,470) | 2e-5 / 2.0 | killed | — | inconclusive | abandon_line |
| exp-02 | 119 | 0.31 | sft | base_model | combined_math_train.jsonl (247,470) | 2e-5 / 2.0 | completed | 0.060 @ n=50 (eval_limit50.json) | inconclusive | adopt |
| exp-03 | 177 | 2.75 | decode-config | exp-02 | — | — | completed | 0.067 @ n=30 (eval_eos_fix.json) | inconclusive | reject |
| exp-04 | 198 | 2.81 | sft | base_model | gsm8k_train_clean.jsonl (7,473) | 2e-5 / 5.0 | completed | 0.250 @ n=40 (eval_gsm40.json) | inconclusive | adopt |
| exp-05 | 260 | 3.23 | sft | exp-04 | fewshot_math_train.jsonl (254,943) | 1e-5 / 1.0 | killed | — | inconclusive | abandon_line |
| exp-06 | 295 | 3.45 | sft | exp-04 | fewshot_math_train.jsonl (254,943) | 1e-5 / 1.5 | killed | — | inconclusive | abandon_line |
| exp-07 | 321 | 3.67 | sft | exp-04 | gsm_fewshot_train.jsonl (14,946) | 1e-5 / 3.0 | completed | 0.540 @ n=50 (eval_fs50.json) | inconclusive | adopt |
| exp-08 | 355 | 4.39 | sft | exp-07 | fewshot_math_train.jsonl (254,943) | 8e-6 / 1.0 | completed | 0.670 @ n=100 (eval_meta100.json) | inconclusive | adopt |
| exp-09 | 411 | 8.19 | sft | exp-08 | fewshot_math_train.jsonl (254,943) | 3e-6 / 0.35 | completed | 0.680 @ n=150 (eval_final150.json), 0.677 @ n=1319 (eval_full.json) | inconclusive | adopt |
| exp-10 | 451 | 9.58 | decode-config | exp-09 | — | — | completed | — | inconclusive | adopt |

Every verdict is `inconclusive` because no two evaluations along the incumbent
chain were taken under the same `--limit` (50, 30, 40, 50, 100, 150/full), so no
card has a comparator measured under its own protocol.

The handed-over artifact is `/home/ben/task/final_model`: exp-09's weights,
packaged by exp-10.
