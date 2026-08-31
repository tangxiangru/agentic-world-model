# r-f099ff8e — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, one H100.
11 cards, one per launch found in the digest. Measurements are the agent's own
evals (myeval.py at temperature 1.0 with the eval's 10-shot system prompt,
unless marked "benchmark evaluator"). Accuracies quoted as fractions.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 188 | 0.29 | sft | base_model | data/gsm8k_train.jsonl | 1e-5 / 3 | completed | 0.1133 @ n=300 (0.70 greedy @ n=100) | inconclusive | reject |
| exp-02 | 341 | 0.66 | sft | base_model | data/sft_mix1.jsonl (gsm8k x2 + 120k metamath_gsm + 40k metamath_math) | 1e-5 / 2 | completed | 0.033 @ n=30 (0.75 greedy @ n=100; 0.583 @ n=60 without the few-shot system prompt) | inconclusive | adopt |
| exp-03 | 497 | 2.57 | sft | exp-02 | data/sft_fs1.jsonl (50k few-shot-augmented) | 7e-6 / 2 | killed | none | inconclusive | abandon_line |
| exp-04 | 551 | 2.76 | sft | exp-02 | data/sft_fs2.jsonl (18k few-shot-augmented) | 1e-5 / 2 (killed at 1 epoch) | killed | 0.595 @ n=200; 0.595 @ n=200 benchmark evaluator | inconclusive | adopt |
| exp-05 | 676 | 3.72 | rft | exp-04 | data/rft1_fs.jsonl (self-samples n=8 + gold, few-shot-augmented) | 1e-5 / 1 | completed | 0.67 @ n=300; 0.65 @ n=300 benchmark evaluator; 0.606 @ n=500 | inconclusive | reject |
| exp-06 | 790 | 5.06 | rft | exp-04 | data/rft12_fs.jsonl (rft1+rft2 samples capped 4/question + gold, few-shot-augmented) | 1e-5 / 1 | completed | 0.632 @ n=500 (+0.026 vs exp-05) | inconclusive | adopt |
| exp-07 | 918 | 6.10 | grpo | exp-06 | gsm8k train prompts (in-memory, few-shot eval condition) | 3e-6 / 160 steps | killed | none | inconclusive | abandon_line |
| exp-08 | 934 | 6.17 | grpo | exp-06 | gsm8k train prompts (in-memory, few-shot eval condition) | 3e-6 / 160 steps | completed | 0.756 @ n=500 (+0.124 vs exp-06); 0.754 benchmark evaluator | supported | adopt |
| exp-09 | 1010 | 7.12 | grpo | exp-08 | gsm8k train prompts (in-memory, few-shot eval condition) | 3e-6 / 120 steps | completed | 0.790 @ n=500 (+0.034 vs exp-08); 0.801 @ n=800 benchmark evaluator on final_model | supported | adopt |
| exp-10 | 1064 | 7.78 | grpo | exp-09 | gsm8k train prompts (in-memory, few-shot eval condition) | 3e-6 / 120 steps | completed | 0.794 @ n=500 (+0.004 vs exp-09) | inconclusive | reject |
| exp-11 | 1152 | 8.60 | merge | exp-09 | none (weight average of exp-09 and exp-10 outputs) | n/a | completed | 0.794 @ n=500 (+0.004 vs exp-09) | inconclusive | reject |

Submission: **exp-09** — `work/grpo2`, copied to `final_model` at [1050] and never
replaced; verified there at 0.801 (stderr 0.014) on 800 items and 0.765 on 200 at
the evaluator's default settings.

Not cards: the baseline eval of the base model at [44] (the comparator, 0.113 at
limit 150), the RFT sampling passes at [638] and [756] (data generation, recorded
in the consuming card's `setup.data`), and the `final_model` copies at [684],
[734], [908], [984], [1050] (promotions of checkpoints that already have cards,
recorded in those cards). Launch attempts that produced no run, and crashed
pipeline steps, are listed as `provenance.smoke_runs` on the card they precede.
`combine_rft.py` was written at [377] but never run.
