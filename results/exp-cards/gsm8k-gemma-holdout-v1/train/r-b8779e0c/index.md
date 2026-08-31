# r-b8779e0c — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
All evals are the agent's own `evaluate.py --limit 150` (official split) unless noted.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 77 | 0.09 | sft | base_model | math_training_data (gsm8k train 1x + MetaMathQA GSM_*, 69399) | 2e-5 / 2 | completed | 0.527 @ n=150 (eval_results.json); 0.360 @ n=50 | inconclusive | adopt |
| exp-02 | 110 | 1.91 | sft | base_model | math_training_data_v2 (gsm8k 3x + MetaMathQA + OpenR1-Math + phrasing variants, 126764) | 3e-5 / 1 | completed | 0.320 @ n=150 (eval_results_v2.json), -0.207 vs exp-01 | contradicted | reject |
| exp-03 | 134 | 3.77 | sft | base_model | math_training_data_v3 (gsm8k 5x + MetaMathQA GSM_*, 99291) | 2e-5 / 3 | completed | 0.580 @ n=150 (eval_results_v3.json), +0.053 vs exp-01 | supported | adopt |
| exp-04 | 158 | 7.16 | sft | base_model | math_training_data_v4 (gsm8k 10x + MetaMathQA capped 30000, 104730) | 1e-5 / 2 | completed | 0.493 @ n=150 (eval_results_v4.json), -0.087 vs exp-03 | contradicted | reject |
| exp-05 | 170 | 9.52 | other (packaging) | exp-03 | — | — / — | completed | 0.593 @ n=150 (stream only, [178]), +0.013 vs exp-03 | inconclusive | adopt (submission) |
