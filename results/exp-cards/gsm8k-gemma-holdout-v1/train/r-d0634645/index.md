# r-d0634645 - reconstructed experiment cards

Base model Qwen/Qwen3-4B-Base, benchmark gsm8k, 10 h on one H100. 18 launches.
`best measurement` is the highest-scoring of the card's own evals, with its sample count;
all accuracies are the run's own `evaluate.py` numbers, never an official score.

| exp-NN | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [68] | 0.07 | sft | base_model | sft_train.jsonl | 2e-05/2 | failed | - | inconclusive | abandon_line |
| exp-02 | [113] | 0.11 | sft | base_model | sft_train.jsonl | 2e-05/2 | completed | 0.08 @50 | inconclusive | adopt |
| exp-03 | [217] | 2.48 | decode-config | exp-02 | - | -/- | completed | 0.62 @50 | supported | adopt |
| exp-04 | [230] | 2.53 | sft | exp-03 | sft_train_v2.jsonl | 1e-05/1.5 | completed | 0.82 @50 | supported | adopt |
| exp-05 | [289] | 3.93 | grpo | exp-04 | sft_train_v2.jsonl | 1e-06/- | failed | - | inconclusive | abandon_line |
| exp-06 | [299] | 3.95 | grpo | exp-04 | sft_train_v2.jsonl | 1e-06/- | killed | - | inconclusive | abandon_line |
| exp-07 | [308] | 3.96 | grpo | exp-04 | sft_train_v2.jsonl | 1e-06/- | killed | - | inconclusive | abandon_line |
| exp-08 | [325] | 3.97 | grpo | exp-04 | sft_train_v2.jsonl | 1e-06/- | failed | - | inconclusive | abandon_line |
| exp-09 | [351] | 4.01 | grpo | exp-04 | sft_train_v2.jsonl | 1e-06/- | completed | 0.6533 @150 | contradicted | reject |
| exp-10 | [399] | 5.02 | sft | exp-04 | sft_train_v3.jsonl | 5e-06/1 | completed | 0.7067 @150 | inconclusive | reject |
| exp-11 | [466] | 6.04 | rft | exp-04 | rft_train.jsonl | 5e-06/2 | completed | 0.7275 @400 | supported | adopt |
| exp-12 | [501] | 7.18 | sft | exp-04 | sft_train_v4.jsonl | 5e-06/2 | failed | 0.6667 @150 | contradicted | reject |
| exp-13 | [536] | 8.34 | sft | exp-04 | sft_mix.jsonl | 3e-06/1 | failed | - | inconclusive | abandon_line |
| exp-14 | [547] | 8.35 | sft | exp-04 | sft_mix.jsonl | 3e-06/1 | completed | 0.66 @400 | inconclusive | reject |
| exp-15 | [557] | 9.15 | other | exp-04 | - | -/- | completed | - | contradicted | reject |
| exp-16 | [567] | 9.20 | other | exp-11 | - | -/- | completed | 0.708 @400 | supported | reject |
| exp-17 | [573] | 9.25 | grpo | exp-11 | rft_train.jsonl | 5e-06/- | completed | 0.7525 @400 | supported | adopt |
| exp-18 | [596] | 9.70 | other | exp-17 | - | -/- | completed | 0.78 @100 | supported | adopt |
