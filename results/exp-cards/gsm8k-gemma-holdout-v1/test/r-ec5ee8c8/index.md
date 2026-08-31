# r-ec5ee8c8 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 123 | null | sft | base_model | openai/gsm8k:main (7473) | 1.5e-4 / 2 | completed | none (adapter not scored) | inconclusive | adopt |
| exp-02 | 153 | null | merge | exp-01 | - | - / - | completed | accuracy 0.480 @ n=100 (expA_100.json) | inconclusive | reject |
| exp-03 | 164 | null | sft | base_model | openai/gsm8k:main+socratic (14946) | 1.2e-4 / 1 | completed | none (adapter not scored) | inconclusive | adopt |
| exp-04 | 197 | null | merge | exp-03 | - | - / - | completed | accuracy 0.520 @ n=150 (expB_150.json) | supported | adopt |
| exp-05 | 223 | null | other (packaging to final_model) | exp-04 | - | - / - | completed | accuracy 0.500 @ n=50 (final_model_50.json) | inconclusive | adopt |
