# r-aaf3560a - reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base | benchmark: gsm8k | budget: 10 h, 1x H100
8 cards. Measurements are the agent's own `evaluate.py` runs; the `--limit` is given with each.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 18 | 0.02 | other (package base) | base_model | - | - | completed | 0.167 @ n=30 | inconclusive | reject |
| exp-02 | 64 | 0.15 | sft | base_model | openai/gsm8k train, 7,473, plain text | 2e-5 / 3 | completed | 0.100 @ n=50 | inconclusive | reject |
| exp-03 | 87 | 0.44 | sft | base_model | openai/gsm8k train, 7,473, chat template | 1e-5 / 2 | completed | 0.200 @ n=50 | supported | reject |
| exp-04 | 107 | 0.61 | sft | base_model | openai/gsm8k train, 22,419 (3x, 4 system prompts) | 5e-6 / 4 | completed | 0.340 @ n=50; 0.300 @ n=100, n=150 | inconclusive | adopt |
| exp-05 | 125 | 1.47 | sft | base_model | openai/gsm8k train, 37,365 (5x, 3 system prompts) | 3e-6 / 5 (3 run) | killed | 0.330 @ n=100 | supported | adopt |
| exp-06 | 144 | 2.53 | sft | exp-05 | openai/gsm8k train, 22,419 (3x, 1 system prompt) | 1e-6 / 2 | completed | 0.250 @ n=100 | contradicted | reject |
| exp-07 | 153 | 2.90 | other (repackage) | exp-05 | - | - | completed | 0.233 @ n=30 | inconclusive | reject |
| exp-08 | 187 | 2.96 | other (repackage) | exp-04 | - | - | completed | 0.340 @ n=50; 0.300 @ n=100, n=150 | inconclusive | adopt |

Submitted: exp-08 - the exp-04 checkpoint copied back into `final_model` at [187]; the run ends
in that state [201]. exp-04 and exp-05 are also `adopt` because each output became the parent of
a later card (exp-05 -> exp-06/exp-07, exp-04 -> exp-08).

Not cards: eight pre-launch crashes recorded as `provenance.smoke_runs` - six TRL/TrainingArguments
API errors before the first real launch (exp-02), one missing chat template (exp-03), one
dataset-map shape error (exp-04).
