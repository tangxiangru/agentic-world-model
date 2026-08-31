# r-736cc5f9 — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on one H100.
The digest carries no timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 110 | null | sft (LoRA r=64) | base_model | gsm8k train, 0-shot targets (eval prompts 10-shot) | 2e-4 / 3 | killed | none | inconclusive | abandon_line |
| exp-02 | 170 | null | sft (LoRA r=64) | base_model | gsm8k train, 0-shot targets | 2e-4 / 3 | completed | 0.60 acc, local test-100 (agent re-parse of exp1_local_test100.json) | inconclusive | reject |
| exp-03 | 335 | null | sft (full fine-tune) | base_model | gsm8k train, 0-shot targets | 5e-5 / 2 | completed | 0.66 acc, local test-100 (exp2_local_test100_stopfix.json); 0.63 official-100 (exp2_eval100_fast.json) | supported | adopt |
| exp-04 | 401 | null | sft (full fine-tune) | base_model | gsm8k train, 4-shot system prompt | 3e-5 / 1 | killed | none (eval_loss 0.3733 at step 150 vs 0.3702) | inconclusive | abandon_line |
| exp-05 | 427 | null | other (packaging) | exp-03 | none | null / null | completed | 0.63 acc, official-100 (exp2_eval100_fast.json, run on the source checkpoint) | inconclusive | adopt |
