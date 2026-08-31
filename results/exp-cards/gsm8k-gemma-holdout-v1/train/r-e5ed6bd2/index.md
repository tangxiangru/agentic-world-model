| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 52 | 0.06 | sft (LoRA r=64) | base_model | gsm8k_train_templated.jsonl (7473) | 2e-4 / 3 | failed | - | inconclusive | abandon_line |
| exp-02 | 64 | 0.07 | sft (LoRA r=64) | base_model | gsm8k_train_templated.jsonl (7473) | 2e-4 / 3 | completed | accuracy 0.027, n=150 (first_iteration_results.json) | inconclusive | reject |
| exp-03 | 118 | 0.50 | sft (LoRA r=32) | base_model | gsm8k_train_templated_v2.jsonl (7473) | 1e-4 / 2 | completed | accuracy 0.020, n=150 (second_iteration_results.json) | contradicted | reject |
| exp-04 | 148 | 0.98 | sft (LoRA r=16, TRL) | base_model | gsm8k_train_simple.jsonl (7473) | 5e-5 / 3 | failed | - | inconclusive | abandon_line |
| exp-05 | 154 | 0.99 | sft (LoRA r=16, TRL) | base_model | gsm8k_train_simple.jsonl (7473) | 5e-5 / 3 | failed | - | inconclusive | abandon_line |
| exp-06 | 160 | 1.00 | sft (LoRA r=16) | base_model | gsm8k_train_simple.jsonl (7473) | 5e-5 / 3 | completed | accuracy 0.173, n=150 (third_iteration_results.json); 0.121, n=1319 (full_evaluation_results.json) | contradicted | reject |
| exp-07 | 191 | 1.78 | sft (full fine-tune) | base_model | gsm8k_train_simple.jsonl (7473) | 1e-5 / 1 | completed | accuracy 0.078, n=1319 (full_ft_results.json) | contradicted | reject |
| exp-08 | 197 | 2.20 | other (packaging) | base_model | - | - / - | completed | accuracy 0.080, n=50 (final_verification.json) | inconclusive | adopt |
