# Reconstructed experiment cards - run r-c1110c15

Base model post-trained: HuggingFaceTB/SmolLM3-3B-Base. Benchmark: gsm8k, 10 h budget, one H100.
Every row cites the event index of its launch in the run digest. Accuracies are the agent's own
evaluate.py runs; no official score is recorded anywhere in these cards.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 1358 | 4.68 | distill | (pre-digest) final_model | gsm8k_teacher_qwen_math7b_direct | 1e-06 / 25 steps | completed | 0.780 @100 (qwen_teacher_direct_refine_a_ckpt5_eval100.json) | contradicted | reject |
| exp-02 | 1378 | 4.83 | distill | (pre-digest) final_model | gsm8k_teacher_qwen_math7b_direct | 5e-07 / 5 steps | completed | 0.790 @100 (qwen_teacher_direct_refine_b_ckpt5_eval100.json) | contradicted | reject |
| exp-03 | 1411 | 4.99 | distill | (pre-digest) final_model | gsm8k_teacher_delta_mix_a | 5e-07 / 15 steps | completed | 0.810 @100 (teacher_delta_mix_a_ckpt5_eval100.json) | supported | adopt |
| exp-04 | 1429 | 5.13 | other | exp-03 | - | - / - | completed | 0.700 @10 (final_model_eval10_latest4.json) | inconclusive | adopt |
| exp-05 | 1511 | 5.26 | distill | exp-04 | gsm8k_teacher_delta_mix_b | 5e-07 / 10 steps | completed | 0.720 @100 (teacher_delta_mix_b_ckpt5_eval100.json) | contradicted | reject |
| exp-06 | 1529 | 5.32 | distill | exp-04 | gsm8k_teacher_delta_mix_c | 5e-07 / 10 steps | completed | 0.780 @100 (teacher_delta_mix_c_ckpt5_eval100.json) | contradicted | reject |
| exp-07 | 1545 | 5.39 | distill | exp-04 | gsm8k_teacher_delta_mix_d | 5e-07 / 10 steps | completed | 0.820 @100 (teacher_delta_mix_d_ckpt5_eval100.json) | contradicted | reject |
| exp-08 | 1598 | 5.56 | sft | exp-04 | gsm8k_gold_delta_mix_a | 5e-07 / 5 steps | completed | 0.690 @100 (gold_delta_mix_a_ckpt5_eval100.json) | contradicted | reject |
| exp-09 | 1614 | 5.63 | distill | exp-04 | gsm8k_teacher_delta_mix_a | 5e-07 / 4 steps | completed | 0.770 @100 (teacher_delta_mix_a_step4_eval100.json) | contradicted | reject |
| exp-10 | 1646 | 5.72 | distill | exp-04 | gsm8k_teacher_delta_mix_a | 5e-07 / 15 steps | completed | 0.700 @100 (teacher_delta_mix_a115_step4_eval100.json) | contradicted | reject |
| exp-11 | 1669 | 5.86 | distill | exp-04 | gsm8k_teacher_delta_mix_a | 2e-07 / 5 steps | completed | 0.710 @100 (teacher_delta_mix_a_lr2e7_ckpt5_eval100.json) | contradicted | reject |
| exp-12 | 1686 | 5.93 | merge | exp-04 | - | - / - | completed | 0.700 @100 (final_model_soup_d20_eval100.json) | contradicted | reject |
| exp-13 | 1714 | 6.01 | decode-config | exp-04 | - | - / - | completed | 0.320 @100 (final_model_fixregex_v2_eval100.json) | contradicted | reject |
| exp-14 | 1757 | 6.12 | dpo | exp-04 | gsm8k_dpo_wrong_final_model | 5e-06 / - | failed | - | inconclusive | abandon_line |
| exp-15 | 1763 | 6.13 | dpo | exp-04 | gsm8k_dpo_wrong_final_model | 5e-06 / - | killed | - | inconclusive | abandon_line |
| exp-16 | 1774 | 6.16 | dpo | exp-04 | gsm8k_dpo_wrong_final_model | 5e-06 / 1 epochs | completed | 0.710 @100 (dpo_wrong_final_a_ckpt20_eval100.json) | contradicted | reject |
| exp-17 | 1888 | 6.37 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_e_overlap_rand | 5e-07 / 15 steps | completed | 0.680 @100 (teacher_delta_mix_e_overlap_rand_ckpt5_eval100.json) | contradicted | reject |
| exp-18 | 1899 | 6.4 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_f_overlap_short | 5e-07 / 15 steps | completed | 0.710 @100 (teacher_delta_mix_f_overlap_short_ckpt10_eval100.json) | contradicted | reject |
| exp-19 | 1928 | 6.49 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_qwen_multisample | 5e-07 / 15 steps | completed | - | inconclusive | abandon_line |
| exp-20 | 1940 | 6.53 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_qwen_multisample | 5e-07 / 15 steps | completed | 0.720 @100 (teacher_delta_mix_a_qwen_multisample_keep3_ckpt5_eval100.json) | contradicted | reject |
| exp-21 | 1987 | 6.65 | decode-config | exp-04 | - | - / - | completed | - | inconclusive | abandon_line |
| exp-22 | 2009 | 6.69 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_seed202 | 5e-07 / 15 steps | completed | 0.710 @100 (teacher_delta_mix_a_seed202_ckpt5_eval100.json) | contradicted | reject |
| exp-23 | 2016 | 6.73 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_seed101 | 5e-07 / 15 steps | killed | - | inconclusive | abandon_line |
| exp-24 | 2038 | 6.8 | merge | exp-04 | - | - / - | failed | - | inconclusive | abandon_line |
| exp-25 | 2039 | 6.8 | merge | exp-04 | - | - / - | failed | - | inconclusive | abandon_line |
| exp-26 | 2062 | 6.83 | merge | exp-04 | - | - / - | completed | 0.740 @100 (final_model_openmath_soup90_eval100.json) | contradicted | reject |
| exp-27 | 2068 | 6.87 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_seed101 | 5e-07 / 15 steps | completed | 0.720 @100 (teacher_delta_mix_a_seed101_ckpt5_eval100.json) | contradicted | reject |
| exp-28 | 2088 | 6.97 | merge | exp-04 | - | - / - | completed | 0.770 @100 (final_model_openmath_extrap105_eval100.json) | contradicted | reject |
| exp-29 | 2089 | 6.97 | merge | exp-04 | - | - / - | completed | 0.680 @100 (final_model_openmath_extrap110_eval100.json) | contradicted | reject |
| exp-30 | 2211 | 7.19 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_seed303 | 5e-07 / 15 steps | completed | - | inconclusive | abandon_line |
| exp-31 | 2221 | 7.22 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_delta_mix_a_seed303 | 5e-07 / 15 steps | completed | 0.770 @100 (teacher_delta_mix_a_seed303_ckpt5_eval100.json) | contradicted | reject |
| exp-32 | 2305 | 7.41 | sft | exp-04 | gsm8k_wrong_gold_mix_a | 2e-07 / 10 steps | completed | 0.820 @100 (wrong_gold_mix_a_ckpt5_eval100.json) | contradicted | reject |
| exp-33 | 2322 | 7.52 | sft | exp-04 | gsm8k_wrong_gold_mix_b | 1e-07 / 10 steps | completed | 0.770 @100 (wrong_gold_mix_b_ckpt5_eval100.json) | contradicted | reject |
| exp-34 | 2336 | 7.6 | sft | exp-04 | gsm8k_teacher_plus_wrong_mix_a | 1e-07 / 10 steps | completed | 0.750 @100 (teacher_plus_wrong_mix_a_ckpt5_eval100.json) | contradicted | reject |
| exp-35 | 2355 | 7.67 | merge | exp-04 | - | - / - | completed | 0.700 @100 (final_wronga_soup90_eval100.json) | contradicted | reject |
| exp-36 | 2393 | 7.78 | distill | (pre-digest) openmath_eval_refine_b_ckpt25_merged | gsm8k_teacher_plus_wrong_mix_a | 5e-07 / 15 steps | completed | 0.720 @100 (teacher_plus_wrong_from_openmath_a_ckpt5_eval100.json) | contradicted | reject |
