# r-dc28e30d — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 147 | 0.22 | sft | base_model | gsm8k train (7473) | 2e-5 / 3 | completed | 0.460 @ n=200 (checkpoint-468) | inconclusive | reject |
| exp-02 | 309 | 0.64 | sft | base_model | gsm8k x3 + MetaMathQA-GSM 80k (102419) | 2e-5 / 2 | failed | none (OOM on first batch) | inconclusive | abandon_line |
| exp-03 | 358 | 0.71 | sft | base_model | gsm8k x3 + MetaMathQA-GSM 80k (102419) | 2e-5 / 2 | completed | 0.630 @ n=200 (runs/sft2) | supported | adopt |
| exp-04 | 483 | 2.41 | other (package to final_model) | exp-03 | — | — / — | completed | 0.630 @ n=200 (inherited from runs/sft2) | inconclusive | adopt |
| exp-05 | 531 | 2.47 | grpo | exp-03 | gsm8k train prompts (7473) | 1e-6 / 0.54 (250 steps) | completed | 0.800 @ n=200 (checkpoint-200); 0.762 @ n=500 | supported | iterate |

Smoke runs folded into cards (not cards themselves): [487] GRPO 3-step pipeline check → exp-05.

Run-level note: the digest ends at t=+3.38h of a 10 h budget, mid-way through the
--limit 500 confirmation evals launched at [617]; the workspace snapshot agrees
(checkpoint-250's l500 log has no .json). final_model was last set at [483] to the
exp-03 checkpoint, and the stream does not show whether a GRPO checkpoint replaced it.
