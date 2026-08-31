| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 49 | null | sft | base_model | gsm8k train (64 ex) | 2e-5 / 0.1 | completed | accuracy 0.05 @ n=20 (runs/smoke_eval20.json) | inconclusive | reject |
| exp-02 | 74 | null | sft | base_model | gsm8k train (7473 ex) | 2e-5 / 1.0 | completed | none (both evals killed) | inconclusive | abandon_line |
| exp-03 | 137 | null | sft | base_model | gsm8k train (7473 ex) | 1.5e-5 / 2.0 | completed | accuracy 0.00 @ n=40 (runs/exp2_eval40.json) | inconclusive | reject |
| exp-04 | 187 | null | other (tokenizer-asset replacement) | exp-03 | none | null / null | completed | none (verification eval killed) | inconclusive | abandon_line |
| exp-05 | 199 | null | sft | base_model | gsm8k train (7473 ex, 0.25 epoch) | 5e-6 / 0.25 | completed | none (80-sample eval never reported) | inconclusive | adopt |
| exp-06 | 211 | null | other (packaging to final_model) | exp-05 | none | null / null | completed | accuracy 0.00 @ n=20 (final_model_eval20.json) | inconclusive | reject |
| exp-07 | 229 | null | sft | base_model | gsm8k train (64 ex) | 1e-7 / 0.05 | completed | none (eval output never written) | inconclusive | adopt |
| exp-08 | 237 | null | other (packaging to final_model) | exp-07 | none | null / null | completed | none (final_model never re-evaluated) | inconclusive | adopt |
