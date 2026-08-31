| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 206 | 2.01 | sft | base_model | data/sft_run1 (OpenMathInstruct-2 train_1M + GSM8K train, decontaminated) | 2e-5 / 1.0 (stopped at step 2000 of 3101) | killed | 94.8% holdout (n=250) at step 2000; 47.3% at --limit 150 through the corrupted server | supported | adopt |
| exp-02 | 472 | 4.73 | decode-config | exp-01 | none | none | completed | 86.9% (n=700), 87.3% (n=150) with the default eval command | supported | adopt |
| exp-03 | 498 | 4.92 | grpo | exp-01 | GSM8K train prompts minus holdout (built in-script) | 2e-6 / 1 (240 steps planned) | killed | none | inconclusive | abandon_line |
| exp-04 | 518 | 5.01 | grpo | exp-01 | GSM8K train prompts minus holdout (built in-script) | 2e-6 / 1 (240 steps) | completed | 85.3% (n=150) vs 87.3% for exp-02; holdout 94.4% | contradicted | reject |
| exp-05 | 642 | 7.91 | other (architecture relabel to llama4) | exp-01 | none | none | completed | none in the stream | inconclusive | reject |
| exp-06 | 702 | 8.03 | other (architecture relabel to llama4, rotary fix) | exp-01 | none | none | completed | none in the stream | inconclusive | abandon_line |
| exp-07 | 746 | 8.20 | sft | exp-01 | data/sft_anneal (comma-formatted answers) | 6e-6 / 1.0 | completed | none in the stream | inconclusive | reject |
| exp-08 | 782 | 8.82 | sft | exp-01 | data/sft_anneal2 (comma-formatted, OpenMathInstruct-style only) | 2.5e-6 / 0.5 | completed | 0 of 31 four-digit holdout answers written with a separator, unchanged from the parent | inconclusive | reject |

Notes

- exp-02 is the submitted model: the step-2000 weights of exp-01, copied to final_model with the serving context capped at 8192.
- Smoke and benchmark runs are not cards: two 25-30 step throughput benchmarks before exp-01 ([194], [200]) and a 3-step GRPO smoke run before exp-03 ([481]) are recorded in the relevant card's provenance.smoke_runs.
- scripts/package_model.py runs ([327], [433], [576], [763]/[771], [792]) apply the same fixed serving config to every candidate before evaluation, so they are treated as part of the evaluation pipeline rather than as their own launches. The one exception is [472], which copied a package to final_model and changed its context cap; that is exp-02.
