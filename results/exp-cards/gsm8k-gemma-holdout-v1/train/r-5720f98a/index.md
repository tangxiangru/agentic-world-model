# Reconstructed experiment cards - r-5720f98a

Base model Qwen/Qwen3-1.7B-Base, benchmark gsm8k, 10 h budget, 1x NVIDIA H100 80GB HBM3.
32 cards, one per launch that can be pointed at in the digest, in launch order.
Accuracies are the run's own `evaluate.py` numbers; official accuracy is deliberately not recorded.

| exp-NN | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 99 | 0.33 | sft | base_model | gsm8k_train | 3e-05 / 4 | failed | - | inconclusive | abandon_line |
| exp-02 | 104 | 0.34 | sft | base_model | gsm8k_train | 3e-05 / 4 | completed | 0.4867 @150 | inconclusive | adopt |
| exp-03 | 152 | 0.52 | sft | base_model | openmath2_gsm_train | 2e-05 / 1 | killed | - | inconclusive | abandon_line |
| exp-04 | 157 | 0.54 | sft | base_model | openmath2_gsm_train | 2e-05 / 1 | completed | 0.2200 @150 | contradicted | adopt |
| exp-05 | 255 | 1.3 | sft | exp-04 | gsm8k_train | 2e-05 / 3 | completed | 0.1200 @150 | contradicted | reject |
| exp-06 | 287 | 1.43 | sft | exp-02 | gsm8k_train_context | 1e-05 / 1 | completed | 0.5530 @150 | supported | adopt |
| exp-07 | 367 | 1.83 | rft | exp-06 | gsm8k_train, self_distilled_nocontext | 8e-06 / 1 | killed | - | inconclusive | abandon_line |
| exp-08 | 371 | 1.85 | rft | exp-06 | gsm8k_train, self_distilled_nocontext | 8e-06 / 1 | completed | 0.6333 @150 | supported | adopt |
| exp-09 | 404 | 2.08 | rft | exp-08 | gsm8k_train, self_distilled_nocontext, self_distilled_v2_nocontext | 5e-06 / 1 | completed | 0.5800 @150 | contradicted | reject |
| exp-10 | 422 | 2.26 | sft | exp-08 | gsm8k_train, openmath2_gsm_train | 4e-06 / 1 | completed | 0.5000 @150 | contradicted | reject |
| exp-11 | 448 | 2.48 | sft | exp-08 | gsm8k_hard_curriculum | 4e-06 / 1 | completed | 0.5460 @1319 | contradicted | reject |
| exp-12 | 475 | 2.62 | sft | exp-08 | orca_math_train | 1e-06 / 1 | killed | - | inconclusive | abandon_line |
| exp-13 | 522 | 2.75 | sft | exp-08 | orca_math_train_clean | 1e-06 / 1 | completed | 0.5792 @1319 | contradicted | reject |
| exp-14 | 610 | 3.14 | merge | exp-08 | - | - | completed | - | inconclusive | abandon_line |
| exp-15 | 769 | 3.96 | other | exp-08 | gsm8k_preferences | 5e-06 / 1 | completed | - | inconclusive | adopt |
| exp-16 | 833 | 4.28 | sft | exp-15 | gsm8k_train_context | 5e-06 / 1 | completed | 0.5340 @1319 | contradicted | reject |
| exp-17 | 846 | 4.4 | dpo | exp-08 | gsm8k_preferences | 5e-06 / 1 | completed | 0.5906 @1319 | supported | adopt |
| exp-18 | 877 | 4.72 | dpo | exp-17 | gsm8k_preferences_dpo | 3e-06 / 1 | completed | - | inconclusive | adopt |
| exp-19 | 891 | 4.94 | sft | exp-18 | gsm8k_train_context | 5e-06 / 1 | completed | 0.5470 @1319 | contradicted | reject |
| exp-20 | 906 | 5.06 | dpo | exp-17 | gsm8k_preferences_dpo | 3e-06 / 0.5 | completed | 0.5840 @1319 | contradicted | reject |
| exp-21 | 968 | 5.27 | grpo | exp-17 | gsm8k_train_context | 5e-06 / - | completed | 0.5747 @1319 | contradicted | reject |
| exp-22 | 1020 | 5.67 | dpo | exp-08 | gsm8k_preferences_context | 5e-06 / 1.0 | failed | - | inconclusive | abandon_line |
| exp-23 | 1025 | 5.69 | dpo | exp-08 | gsm8k_preferences_context | 5e-06 / 1.0 | completed | - | inconclusive | abandon_line |
| exp-24 | 1117 | 6.6 | dpo | exp-08 | gsm8k_preferences | 5e-06 / 0.5 | completed | - | inconclusive | abandon_line |
| exp-25 | 1131 | 6.75 | dpo | exp-08 | gsm8k_preferences | 3e-06 / 1.0 | completed | 0.5876 @1319 | contradicted | reject |
| exp-26 | 1163 | 7.04 | dpo | exp-08 | gsm8k_preferences | 5e-06 / 1 | completed | 0.5830 @1319 | contradicted | reject |
| exp-27 | 1235 | 7.49 | dpo | exp-08 | gsm8k_preferences_local8 | 5e-06 / 1 | completed | - | inconclusive | abandon_line |
| exp-28 | 1258 | 7.63 | sft | exp-17 | gsm8k_train_context | 1e-06 / 1 | completed | 0.5750 @1319 | contradicted | reject |
| exp-29 | 1270 | 7.72 | dpo | exp-08 | gsm8k_preferences_hybrid8 | 5e-06 / 1 | completed | 0.5790 @1319 | contradicted | reject |
| exp-30 | 1290 | 7.94 | dpo | exp-08 | gsm8k_preferences_hybrid8_q25 | 5e-06 / 1 | completed | 0.5750 @1319 | contradicted | reject |
| exp-31 | 1303 | 8.15 | other | exp-17 | - | - | completed | 0.5845 @1319 | inconclusive | adopt |
| exp-32 | 1363 | 8.54 | other | exp-17 | - | - | completed | 0.5921 @1319 | supported | adopt |

Adopted / submitted: **exp-32** packages candidates/gsm_ctx_star_dpo (exp-17) into `final_model`;
the same weights are also the output of exp-31, which exp-32 repackaged with the confirmed score.
