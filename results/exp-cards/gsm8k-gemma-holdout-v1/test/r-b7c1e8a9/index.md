| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 77 | null | sft (LoRA r=64) | base_model | train_data.jsonl (gsm8k 7.5K + MetaMathQA-GSM 240K = 247,473) | 2e-4 / 2 | failed | none | inconclusive | abandon_line |
| exp-02 | 88 | null | sft (LoRA r=64) | base_model | train_data.jsonl (247,473) | 2e-4 / 2 | killed | none | inconclusive | abandon_line |
| exp-03 | 107 | null | sft (LoRA r=128) | base_model | train_data.jsonl (247,473) | 2e-4 / 1 | completed | accuracy 0.540 (n=150) | inconclusive | adopt |
| exp-04 | 223 | null | sft (LoRA r=128) | base_model | train_data_v2.jsonl (gsm8k x5 + MetaMathQA-GSM = 277,362) | 1.5e-4 / 2 | killed | none | inconclusive | abandon_line |
| exp-05 | 236 | null | sft (full FT) | base_model | gsm8k x5 + MetaMathQA-GSM, built in-script (count not printed) | 2e-5 / 1 | completed | accuracy 0.113 (n=150) | contradicted | reject |
| exp-06 | 292 | null | sft (LoRA r=64) | exp-03 | gsm8k train only, repeated 8x (count not printed) | 5e-5 / 1 | completed | accuracy 0.153 (n=150) | contradicted | reject |
