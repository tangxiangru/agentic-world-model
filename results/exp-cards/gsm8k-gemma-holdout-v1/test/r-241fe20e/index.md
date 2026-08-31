# r-241fe20e — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on 1x H100 80GB
7 cards. The digest carries no timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 109 | null | sft | base_model | gsm8k main/train (7473) | 2e-4 / 2.0 | failed | none | inconclusive | abandon_line |
| exp-02 | 118 | null | sft | base_model | gsm8k main/train (7473) | 2e-4 / 2.0 | killed | none | inconclusive | abandon_line |
| exp-03 | 137 | null | sft | base_model | gsm8k main/train (7473) | 2e-4 / 2.0 | completed | 0.150 (n=40) | supported | reject |
| exp-04 | 222 | null | sft | base_model | gsm8k main/train (7473), fewshot_system_k=3 | 1.5e-4 / 1.0 | completed | 0.3667 (n=150); 0.325 (n=40) | supported | adopt |
| exp-05 | 255 | null | sft | base_model | gsm8k main/train (7473), fewshot_system_k=3 | 1.2e-4 / 1.5 | killed | none | inconclusive | abandon_line |
| exp-06 | 286 | null | merge | exp-05 | none | null / null | completed | 0.300 (n=40) | contradicted | reject |
| exp-07 | 301 | null | other | exp-04 | none | null / null | completed | 0.325 (n=40) | supported | adopt |

exp-07 is the submission: final_model is a `cp -a` of exp-04's merged checkpoint,
re-evaluated at 0.325 on the 40-sample fast eval (final_model_eval40.json).

Smoke runs folded into cards, not counted above: [88] and [99] on exp-01
(training pipeline), [182] and [201] on exp-03 (1-sample vLLM eval probes).
