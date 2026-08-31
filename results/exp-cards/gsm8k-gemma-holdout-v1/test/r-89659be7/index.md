# r-89659be7 — extracted experiment cards

Base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100.
The digest carries no timestamps, so every `elapsed_h` is null. All eight cards
are one line of SFT: nothing else (RFT, DPO, merging, decode changes) was ever
launched. Every measurement is the agent's own `evaluate.py --limit 50` run
against the official test split; no larger-limit and no baseline-model eval
exists anywhere in the run.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 90 | null | sft | base_model | train_data.jsonl (50K of 247,473: gsm8k train + MetaMathQA GSM_*) | 2e-5 / 2 | failed (died at step 3125 of 6250) | none | inconclusive | adopt |
| exp-02 | 169 | null | sft | exp-01 (sft_output/checkpoint-3000) | train_data.jsonl (same 50K, resumed) | 2e-5 / 2 | completed | accuracy 0.46 (n=50) | inconclusive | iterate |
| exp-03 | 235 | null | sft | base_model | train_data.jsonl (all 247,473) | 2e-5 / 1 | killed (projected 5 h, budget) | none | inconclusive | abandon_line |
| exp-04 | 260 | null | sft | base_model | train_data.jsonl (100K of 247,473) | 2e-5 / 2 | killed (projected 3.3 h, budget) | none | inconclusive | abandon_line |
| exp-05 | 281 | null | sft | base_model | train_data.jsonl (50K of 247,473) | 2e-5 / 2 | completed | accuracy 0.06 (n=50, −0.40) | contradicted | reject |
| exp-06 | 395 | null | sft | base_model | train_data_v4.jsonl (97,419: gsm8k train ×3 + 75K MetaMathQA, "####" stripped) | 2e-5 / 1 | completed | accuracy 0.22 (n=50, −0.24) | contradicted | reject |
| exp-07 | 490 | null | sft | base_model | train_data.jsonl (50K of 247,473) | 2e-5 / 1 | completed | accuracy 0.26 (n=50, −0.20) | contradicted | adopt |
| exp-08 | 532 | null | sft | base_model | train_data.jsonl (50K of 247,473) | 2e-5 / 2 | killed (log ends at step 5559 of 6250, no save) | none | inconclusive | abandon_line |

Notes on the two `adopt`s: exp-01's checkpoint-3000 is the parent of exp-02, and
exp-07's weights are the last ones the stream shows written to `final_model` —
exp-08 was launched to replace them and never saved. The best measurement of the
run, exp-02's 0.46, was overwritten in place at launch 281 and cannot be
recovered from the workspace.
