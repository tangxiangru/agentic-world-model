| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 453 | 1.37 | sft | base_model (runs/sft_candidates/step120) | metamath_ansaug_train.jsonl (79847) | 3e-6 / 1 ep (stopped at step 150) | killed | 0.781 @64 | contradicted | adopt |
| exp-02 | 520 | 1.63 | sft | exp-01 (runs/sft_ansaug_v3/checkpoint-150) | metamath_ansaug_train.jsonl (79847) | 3e-6 / 1 ep (stopped at step 300) | killed | 0.827 @150 | contradicted | reject |
| exp-03 | 571 | 1.86 | sft | base_model (runs/sft_candidates/step120) | gsm8k_train_eval_context.jsonl (7473) | 1e-6 / 1 ep (stopped at step 20) | killed | 0.820 @150 | contradicted | reject |
| exp-04 | 637 | 2.17 | grpo | base_model (runs/sft_candidates/step120) | metamath_rephrased_train.jsonl (79888) | 5e-6 / 50 steps | completed | none (adapters scored in exp-05) | inconclusive | reject |
| exp-05 | 658 | 2.30 | merge | base_model (runs/sft_candidates/step120) + exp-04 adapters | none (weight merge) | n/a | completed | 0.827 @150 (step50) | contradicted | reject |
| exp-06 | 693 | 2.51 | grpo | base_model (runs/sft_candidates/step120) | metamath_rephrased_train.jsonl (79888) | 5e-6 / 100 steps (stopped ~75) | killed | none (adapters scored in exp-09) | inconclusive | reject |
| exp-07 | 727 | 2.57 | merge | base_model (runs/sft_candidates/step120) | none (weight interpolation with exp-02) | n/a | completed | none | inconclusive | abandon_line |
| exp-08 | 731 | 2.58 | merge | base_model (runs/sft_candidates/step120) | none (weight interpolation with exp-02) | n/a | completed | 0.867 @150; 0.835 @1319 | supported | adopt |
| exp-09 | 785 | 2.80 | merge | base_model (runs/sft_candidates/step120) + exp-06 adapters | none (weight merge) | n/a | completed | 0.797 @64 | contradicted | reject |
| exp-10 | 794 | 2.83 | merge | base_model (runs/sft_candidates/step120) | none (weight interpolation with exp-02) | n/a | completed | 0.847 @150 (a=0.625) | contradicted | reject |
| exp-11 | 838 | 3.06 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_a075_t09_n4.jsonl (26316) | 1e-6 / 100 steps | completed | 0.836 @1319 (step25) | supported | reject |
| exp-12 | 878 | 3.30 | merge | exp-08 (runs/interp_ansaug_a075) | none (weight interpolation with exp-11) | n/a | completed | 0.860 @150 | contradicted | reject |
| exp-13 | 884 | 3.35 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_balanced_n2.jsonl (21693) | 1e-6 / 75 steps (best at 25) | completed | 0.839 @1319 (1106/1319) | supported | adopt |
| exp-14 | 919 | 3.55 | merge | exp-08 (runs/interp_ansaug_a075) | none (weight interpolation with exp-13) | n/a | completed | 0.847 @150 | contradicted | reject |
| exp-15 | 930 | 3.60 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_hard_weighted.jsonl (18298) | 7.5e-7 / 50 steps | completed | 0.853 @150 | contradicted | reject |
| exp-16 | 932 | 3.61 | merge | base_model (runs/sft_candidates/step120) | none (weight interpolation with exp-02) | n/a | completed | 0.853 @150 (a=0.70) | contradicted | reject |
| exp-17 | 950 | 3.74 | sft | base_model (runs/sft_candidates/step120) | metamath_rephrased_train.jsonl (79888) | 3e-6 / 200 steps (stopped at 100) | killed | 0.840 @150 | supported | reject |
| exp-18 | 973 | 3.89 | merge | base_model (runs/sft_candidates/step120) | none (weight interpolation with exp-17) | n/a | completed | 0.820 @150 | contradicted | reject |
| exp-19 | 979 | 3.93 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_balanced_n2.jsonl (21693) | 1e-6 / 75 steps (stopped at 25) | killed | 0.837 @1319 (1104/1319) | contradicted | reject |
| exp-20 | 996 | 4.01 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (weight interpolation with exp-19) | n/a | completed | 0.847 @150 | contradicted | reject |
| exp-21 | 1034 | 4.10 | sft | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | orca_math_safe.jsonl (199218) | 1e-6 / 75 steps | failed (OOM before step 1) | none | inconclusive | abandon_line |
| exp-22 | 1047 | 4.16 | sft | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | orca_math_safe.jsonl (199218) | 1e-6 / 75 steps (stopped at 25) | killed | 0.840 @150 | contradicted | reject |
| exp-23 | 1077 | 4.36 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (weight interpolation with exp-22) | n/a | completed | 0.836 @1319 (1103/1319) | contradicted | reject |
| exp-24 | 1104 | 4.51 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_balanced_official2_n2.jsonl (29166) | 1e-6 / 75 steps (killed at 33, ckpt 25) | killed | 0.853 @150 | contradicted | reject |
| exp-25 | 1115 | 4.60 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_balanced_n1.jsonl (count not printed) | 1e-6 / 75 steps (killed at 29, ckpt 25) | killed | 0.853 @150 | contradicted | reject |
| exp-26 | 1130 | 4.78 | rft | exp-08 (runs/interp_ansaug_a075) | self_rft_round2_balanced_n2.jsonl (count not printed) | 1e-6 / 75 steps (killed at 30, ckpt 25) | killed | 0.853 @150 | contradicted | reject |
| exp-27 | 1143 | 4.87 | decode-config | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (chat-template edit) | n/a | completed | 0.860 @150 | contradicted | reject |
| exp-28 | 1175 | 5.02 | dpo | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | dpo_preferences_train.jsonl (1939 pairs) | 5e-6, beta 0.1 / 75 steps (killed at 30, ckpt 25) | killed | none (adapter scored in exp-29) | inconclusive | reject |
| exp-29 | 1185 | 5.09 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) + exp-28 adapter | none (weight merge) | n/a | completed | 0.827 @150 | contradicted | reject |
| exp-30 | 1190 | 5.14 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (weight interpolation with exp-29) | n/a | completed | 0.853 @150 | contradicted | reject |
| exp-31 | 1222 | 5.39 | distill | exp-08 (runs/interp_ansaug_a075) | qwen25math7b_teacher_balanced.jsonl (count not printed) | 1e-6 / 75 steps (stopped at 25) | killed | 0.847 @150 | contradicted | reject |
| exp-32 | 1232 | 5.47 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (task vector exp-08 -> exp-31) | n/a | completed | 0.860 @150 | contradicted | reject |
| exp-33 | 1236 | 5.51 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.1) | n/a | completed | 0.8393 @1319 (1107/1319) | supported | reject |
| exp-34 | 1245 | 5.56 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.05) | n/a | completed | 0.8385 @1319 (1106/1319) | contradicted | reject |
| exp-35 | 1253 | 5.63 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.15) | n/a | completed | 0.8415 @1319 (1110/1319); repeat 1105/1319 | supported | adopt |
| exp-36 | 1270 | 5.69 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.20) | n/a | completed | 0.8378 @1319 (1105/1319) | contradicted | reject |
| exp-37 | 1280 | 5.75 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.125) | n/a | completed | 0.8362 @1319 (1103/1319) | contradicted | reject |
| exp-38 | 1284 | 5.81 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.175) | n/a | completed | 0.8355 @1319 (1102/1319) | contradicted | reject |
| exp-39 | 1289 | 5.86 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.14) | n/a | completed | 0.8378 @1319 (1105/1319) | contradicted | reject |
| exp-40 | 1291 | 5.92 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.145) | n/a | completed | 0.8370 @1319 (1104/1319) | contradicted | reject |
| exp-41 | 1295 | 5.97 | merge | exp-13 (runs/sft_balanced_rft_v6/checkpoint-25) | none (Orca task vector, alpha=0.155) | n/a | completed | 0.8385 @1319 (1106/1319) | contradicted | reject |
| exp-42 | 1314 | 6.08 | merge | exp-35 (runs/interp_orca_a015) | none (task vector exp-08 -> exp-26, scale 0.05) | n/a | completed | 0.8324 @1319 (1098/1319) | contradicted | reject |
| exp-43 | 1336 | 6.15 | sft | exp-35 (runs/interp_orca_a015) | self_rft_balanced_n2.jsonl (21693) | 2e-7 / 10 steps | completed | 0.8393 @1319 (1107, repeat 1108) | inconclusive | adopt |
| exp-44 | 1363 | 6.33 | merge | exp-35 (runs/interp_orca_a015) | none (weight interpolation with exp-43) | n/a | completed | 0.8423 @1319 (1111/1319, repeated) | supported | adopt |
| exp-45 | 1375 | 6.43 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.25) | n/a | completed | 0.8362 @1319 (1103/1319) | contradicted | reject |
| exp-46 | 1380 | 6.48 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.75) | n/a | completed | 0.8423 @1319 (1111/1319); repeat 1105/1319 | inconclusive | reject |
| exp-47 | 1392 | 6.58 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.4) | n/a | completed | 0.8378 @1319 (1105/1319) | contradicted | reject |
| exp-48 | 1396 | 6.63 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.6) | n/a | completed | 0.8355 @1319 (1102/1319) | contradicted | reject |
| exp-49 | 1402 | 6.69 | sft | exp-35 (runs/interp_orca_a015) | self_rft_balanced_n2.jsonl (21693) | 2e-7 / 5 steps (seed 20260716) | completed | none (scored via its 0.5 average) | inconclusive | reject |
| exp-50 | 1407 | 6.71 | merge | exp-35 (runs/interp_orca_a015) | none (0.5 blend with exp-49) | n/a | completed | 0.8362 @1319 (1103/1319) | contradicted | reject |
| exp-51 | 1413 | 6.77 | sft | exp-35 (runs/interp_orca_a015) | self_rft_balanced_n2.jsonl (21693) | 2e-7 / 5 steps (seed 424242) | completed | none (scored via its 0.5 average) | inconclusive | reject |
| exp-52 | 1416 | 6.79 | merge | exp-35 (runs/interp_orca_a015) | none (0.5 blend with exp-51) | n/a | completed | 0.8347 @1319 (1101/1319) | contradicted | reject |
| exp-53 | 1422 | 6.84 | sft | exp-44 (runs/interp_polish_a05) | gsm8k_train_all.jsonl (count not printed) | 1e-7 / 5 steps | completed | 0.8408 @1319 (1109/1319) | contradicted | reject |
| exp-54 | 1430 | 6.91 | merge | exp-44 (runs/interp_polish_a05) | none (0.5 blend with exp-53) | n/a | completed | 0.8408 @1319 (1109/1319) | contradicted | reject |
| exp-55 | 1459 | 7.01 | decode-config | exp-44 (runs/interp_polish_a05) | none (chat-template suffix) | n/a | completed | 0.8400 @1319 (1108/1319) | contradicted | reject |
| exp-56 | 1489 | 7.12 | sft | exp-44 (runs/interp_polish_a05) | openr1_word_safe.jsonl (11113) | 1e-6 / 15 steps | failed (discarded and relaunched) | none | inconclusive | abandon_line |
| exp-57 | 1496 | 7.16 | sft | exp-44 (runs/interp_polish_a05) | openr1_word_safe.jsonl (11113) | 1e-6 / 15 steps | completed | none (scored via its blends) | inconclusive | reject |
| exp-58 | 1501 | 7.19 | merge | exp-44 (runs/interp_polish_a05) | none (OpenR1 task vector, alpha=0.25) | n/a | completed | 0.8400 @1319 (1108/1319) | contradicted | reject |
| exp-59 | 1507 | 7.24 | merge | exp-44 (runs/interp_polish_a05) | none (OpenR1 task vector, alpha=0.10) | n/a | completed | 0.8385 @1319 (1106/1319) | contradicted | reject |
| exp-60 | 1516 | 7.30 | other (packaging) | exp-44 (runs/interp_polish_a05) | none (copy to final_model) | n/a | completed | 0.8423 @1319 (1111/1319) | supported | adopt |
| exp-61 | 1527 | 7.38 | decode-config | exp-60 (final_model) | none (fp16 inference dtype) | n/a | completed | 0.8453 @1319 (1115; repeats 1111, 1108) | contradicted | reject |
| exp-62 | 1545 | 7.49 | decode-config | exp-60 (final_model) | none (fp32 inference dtype) | n/a | completed | 0.8370 @1319 (1104/1319) | contradicted | reject |
| exp-63 | 1569 | 7.61 | merge | exp-44 (runs/interp_polish_a05) | none (teacher task vector, scale=0.025) | n/a | completed | 0.8393 @1319 (1107/1319) | contradicted | reject |
| exp-64 | 1574 | 7.66 | merge | exp-44 (runs/interp_polish_a05) | none (teacher task vector, scale=-0.025) | n/a | completed | 0.8370 @1319 (1104/1319) | contradicted | reject |
| exp-65 | 1581 | 7.73 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.49) | n/a | completed | 0.8415 @1319 (1110/1319) | contradicted | reject |
| exp-66 | 1585 | 7.78 | merge | exp-35 (runs/interp_orca_a015) | none (polish blend, alpha=0.51) | n/a | completed | 0.8340 @1319 (1100/1319) | contradicted | reject |
