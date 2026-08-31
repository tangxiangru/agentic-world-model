# r-6dabe99e — reconstructed experiment cards

base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 70 | 0.11 | sft | base_model | gsm8k-train + MetaMathQA GSM_* (247,468 rows) | 2e-5 / 2 | completed | 0.080 @ n=50 (eval_stage1_eos_50.json) | inconclusive | adopt |
| exp-02 | 231 | 3.14 | decode-config | exp-01 | — | — / — | completed | 0.000 @ n=50 (eval_stage1_50.json) | inconclusive | reject |
| exp-03 | 253 | 3.20 | decode-config | exp-01 | — | — / — | completed | 0.080 @ n=50 (eval_stage1_eos_50.json) | supported | reject |
| exp-04 | 262 | 3.24 | sft | exp-01 | gsm8k-train, eval prompt shape | 8e-6 / 4 | completed | 0.460 @ n=150 (eval_stage2_150.json) | supported | adopt |
| exp-05 | 302 | 4.38 | decode-config | exp-04 | — | — / — | completed | 0.400 @ n=50 (eval_stage2_50.json) | supported | adopt |
| exp-06 | 314 | 4.43 | sft | base_model | gsm8k-train, plain prompt shape | 2e-5 / 6 | completed | 0.020 @ n=50 (eval_base_gsm_plain_50.json) | contradicted | reject |
| exp-07 | 334 | 4.73 | decode-config | exp-06 | — | — / — | completed | 0.020 @ n=50 (eval_base_gsm_plain_50.json) | contradicted | reject |
| exp-08 | 356 | 4.76 | sft | base_model | gsm8k-train, eval prompt shape | 8e-6 / 4 | killed | — | inconclusive | abandon_line |
| exp-09 | 391 | 4.90 | decode-config | exp-04 | — | — / — | completed | — (no new eval) | inconclusive | adopt |
| exp-10 | 425 | 4.97 | sft | base_model | cleaned MetaMathQA GSM-derived + gsm8k-train (247,466 rows) | 2e-5 / 2 | completed | — (never evaluated) | inconclusive | adopt |
| exp-11 | 535 | 7.79 | sft | exp-10 | gsm8k-train, eval prompt shape, think format | 5e-6 / 2 | completed | 0.420 @ n=150 (eval_stage2_clean_think_150.json) | contradicted | reject |
| exp-12 | 561 | 8.32 | decode-config | exp-11 | — | — / — | completed | 0.420 @ n=50 (eval_stage2_clean_think_50.json) | contradicted | reject |
| exp-13 | 579 | 8.38 | sft | exp-10 | gsm8k-train, eval prompt shape, plain format | 8e-6 / 4 | completed | 0.480 @ n=50 (eval_stage2_clean_plain4_50.json) | inconclusive | adopt |
| exp-14 | 629 | 9.42 | decode-config | exp-13 | — | — / — | completed | 0.480 @ n=50 (eval_stage2_clean_plain4_50.json) | inconclusive | adopt |

Not cards: the base-model baseline eval at [42] (`evaluate.py --limit 20`, accuracy 0.15 in `baseline_20.json`) is used as the `base_model` comparator; the truncated pipeline check at [66] (16 examples, `--num-train-epochs 0.01`) is recorded as `provenance.smoke_runs` on exp-01.

Final deliverable: `final_model` = the exp-13 weights, packaged by exp-14 [671].
