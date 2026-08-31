# Reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 24351 | 0.36 | sft | base_model | sft_round1.jsonl (167k: gsm8k train + MetaMathQA GSM_* + Orca-Math) | 2e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-02 | 29086 | 0.44 | sft | base_model | sft_round1.jsonl (167k) | 2e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-03 | 32052 | 0.82 | sft | base_model | sft_round1.jsonl (167k) | 2e-5 / 1.0 | completed | 0.6267 @150 (checkpoint-1600); 0.6194 full | inconclusive | adopt |
| exp-04 | 43357 | 4.89 | rft | exp-03 (sft_r1/checkpoint-1600) | sft_round2.jsonl (86.5k: 46.9k rejection-sampled + 39.7k replay) | 1e-5 / 1.0 (70M-token cap) | failed | none | inconclusive | abandon_line |
| exp-05 | 45393 | 5.75 | rft | exp-03 (sft_r1/checkpoint-1600) | sft_round2.jsonl (86.5k) | 1e-5 / 1.0 (70M-token cap) | completed | 0.6657 full (sft_r2/final); 0.673 @150 (checkpoint-950) | supported | adopt |
| exp-06 | 46893 | 7.99 | other (packaging + decode config) | exp-05 (sft_r2/checkpoint-950) | none | n/a | completed | 0.6596 full | contradicted | reject |
| exp-07 | 47771 | 8.28 | other (packaging + decode config) | exp-05 (sft_r2/final) | none | n/a | completed | 0.6933 @150 packaged dir; 0.6657 full | supported | adopt |
