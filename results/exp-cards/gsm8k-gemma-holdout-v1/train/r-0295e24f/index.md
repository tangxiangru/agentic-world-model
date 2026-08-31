| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 129 | null | sft (full) | base_model | gsm8k train, 0-shot, gold rationales | 1e-5 / 3 | completed | 0.047 @ n=150 | inconclusive | adopt |
| exp-02 | 187 | null | other (repackage, fresh tokenizer) | exp-01 | none | null / null | completed | none | inconclusive | abandon_line |
| exp-03 | 206 | null | decode-config (im_end stop fix) | exp-01 | none | null / null | completed | 0.14 @ n=50 | contradicted | reject |
| exp-04 | 234 | null | sft (LoRA r=64) | base_model | gsm8k train, 0-shot, gold rationales | 1e-4 / 3 | killed | none | inconclusive | abandon_line |
| exp-05 | 249 | null | sft (LoRA r=64) | base_model | gsm8k train, 3-shot system prompt, gold rationales | 5e-5 / 1 | completed | 0.04 @ n=50 | contradicted | reject |
| exp-06 | 312 | null | sft (LoRA r=64, near no-op) | base_model | gsm8k train, 128 rows, 1 step | 1e-6 / 0.025 | completed | 0.44 @ n=50 and 0.50 @ n=20 | supported | adopt |
| exp-07 | 411 | null | rft (LoRA r=16) | base_model | data/selftrain_greedy_200.jsonl, 176 self-generated correct traces | 1e-5 / 1 | completed | 0.42 @ n=50 | contradicted | reject |
| exp-08 | 443 | null | rft (LoRA r=8) | base_model | data/selftrain_greedy_1000.jsonl, 893 self-generated correct traces | 5e-6 / 1 | completed | 0.36 @ n=50 | contradicted | reject |
| exp-09 | 459 | null | other (copy to final_model) | exp-06 | none | null / null | completed | 0.50 @ n=20 | supported | adopt |
