# r-45dd80cf — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 95 | 0.22 | sft | base_model | openai/gsm8k train (eval_fixed 10-shot) | 2e-4 / 2 | completed | 0.447 acc @ n=150 | inconclusive | adopt |
| exp-02 | 169 | 1.86 | sft | base_model | openai/gsm8k train (mixed prompts) + MU-NLPC/Calc-gsm8k + ChilleD/SVAMP + MU-NLPC/Calc-mawps (train+val), 42,623 rows | 1.5e-4 / 1 | completed | 0.333 acc @ n=150 | contradicted | reject |
| exp-03 | 220 | 4.16 | sft | base_model | openai/gsm8k train (eval_fixed 10-shot), 7,473 rows | 1.5e-4 / 3 | completed | 0.440 acc @ n=150 | inconclusive | reject |
| exp-04 | 260 | 6.46 | other (packaging) | exp-01 | — | — / — | completed | 0.480 acc @ n=150 | inconclusive | adopt |

Submitted: **exp-04** — `final_model`, a byte-for-byte copy of exp-01's merged
checkpoint, verified at 0.480 on `evaluate.py --limit 150`.

Smoke runs (not cards, recorded on exp-01): launches 69, 75, 78, 84, 87.
