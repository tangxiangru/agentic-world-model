# r-f5bfab57 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 58 | 0.14 | sft | base_model | openai/gsm8k main:train (7473) | 2e-5 / 3 | killed | none | inconclusive | abandon_line |
| exp-02 | 60 | 0.17 | sft | base_model | openai/gsm8k main:train (7473) | 2e-5 / 3 | completed | 0.590 @100 (metrics_100_fixed.json, after eos patch; 0.030 before) | inconclusive | reject |
| exp-03 | 88 | 0.81 | sft | base_model | openai/gsm8k main:train (7473) | 2e-5 / 3 | completed | 0.600 @50 (metrics_50_nothink.json) | inconclusive | reject |
| exp-04 | 106 | 1.21 | sft | base_model | openai/gsm8k main:train (7473) + 10-shot system prompt | 2e-5 / 3 | completed | 0.730 @100 (metrics_100_final.json); 0.6361 @1319 (metrics_full.json) | supported | reject |
| exp-05 | 120 | 2.54 | sft | base_model | openai/gsm8k main:train (7473) + 10-shot system prompt | 2e-5 / 5 | completed | 0.6581 @1319 (metrics_full_5ep.json) | supported | adopt |

Smoke runs (not cards, recorded on exp-01): [32] trl import error, [36] SFTTrainer
kwarg error, [40] passed at 10 steps.

Every launch trains from the base model; no card is the parent of another, and
each training run overwrote `/home/ben/task/final_model` in place.
