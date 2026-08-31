# r-4aa3d061 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 164 | null | sft | base_model | openai/gsm8k main:train (10 fixed few-shots in system, eval-format prompts) | 1e-5 / 3 requested, 2 run | killed | accuracy 0.553 @ n=150 (checkpoint-1832) | inconclusive | adopt |
| exp-02 | 348 | null | other (packaging) | exp-01 | none | null / null | completed | accuracy 0.350 @ n=20 (final_model) | supported | adopt |
