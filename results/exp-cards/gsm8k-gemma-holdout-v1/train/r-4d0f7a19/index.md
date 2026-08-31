# r-4d0f7a19 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 98 | 0.17 | sft | base_model | MetaMathQA GSM-subset 100k + gsm8k-train (generic x1, exact-eval-format x2) | 1e-5 / 1 | completed | accuracy 0.567, n=150 (stage1_150.json) | inconclusive | adopt |
| exp-02 | 227 | 2.25 | sft | exp-01 | gsm8k-train only, 52,311 examples (generic x1, exact-eval-format x6) | 5e-6 / 1 | completed | accuracy 0.540, n=150 (stage2_150.json) | contradicted | reject |
| exp-03 | 293 | 4.40 | sft | base_model | MetaMathQA all 240k GSM rows + gsm8k-train (generic x1, exact-eval-format x2) | 1e-5 / 1 | completed | accuracy 0.527, n=150 (fullgsm_150.json) | contradicted | reject |
| exp-04 | 414 | 8.29 | other (packaging into final_model) | exp-01 | none | n/a | completed | accuracy 0.600, n=150 (final_150.json) | supported | adopt |
