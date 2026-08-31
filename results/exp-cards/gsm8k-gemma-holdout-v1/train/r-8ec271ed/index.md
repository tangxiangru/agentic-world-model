# r-8ec271ed — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
The digest carries no timestamps, so every `elapsed_h` is null.
Base-model reference: accuracy 0.45 on `--limit 40` (baseline_40.json).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 94 | null | sft | base_model | data/gsm8k_train_sft.jsonl (pre-rendered text, n not printed) | 2e-4 / 2.0 (68 steps) | completed | none | inconclusive | adopt |
| exp-02 | 121 | null | merge | exp-01 | – | – | completed | none (v1 line reported as 0.1875 at [275], unassignable) | inconclusive | reject |
| exp-03 | 122 | null | merge | exp-01 | – | – | completed | none (v1 line reported as 0.1875 at [275], unassignable) | inconclusive | reject |
| exp-04 | 150 | null | sft | base_model | data/gsm8k_train_sft.jsonl (messages, n=6473) | 2e-4 / 2.0, --assistant-only-loss | failed | none | inconclusive | abandon_line |
| exp-05 | 154 | null | sft | base_model | data/gsm8k_train_sft.jsonl (messages, n=6473) | 2e-4 / 2.0 (68 steps) | completed | none | inconclusive | adopt |
| exp-06 | 166 | null | merge | exp-05 | – | – | completed | none (eval force-stopped at [185]) | inconclusive | abandon_line |
| exp-07 | 167 | null | merge | exp-05 | – | – | completed | accuracy 0.025, n=40 (max_tokens 400) | contradicted | reject |
| exp-08 | 219 | null | sft | base_model | data/gsm8k_train_sft.jsonl (messages, n=6473) | 1e-6 / 1.0, max_steps 20 | completed | none | inconclusive | adopt |
| exp-09 | 222 | null | merge | exp-08 | – | – | completed | accuracy 0.40, n=40 | contradicted | reject |
| exp-10 | 235 | null | sft | base_model | data/gsm8k_train_sft.jsonl (messages, n=6473) | 1e-6 / 1.0, max_steps 5 | completed | none | inconclusive | adopt |
| exp-11 | 237 | null | merge | exp-10 | – | – | completed | accuracy 0.475, n=40 (best of run) | supported | adopt |
| exp-12 | 246 | null | other (package to final_model) | exp-11 | – | – | completed | none (limit-150 verification killed at [260]) | inconclusive | adopt |
