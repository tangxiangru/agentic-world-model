# Reconstructed experiment cards

19 cards, one per launch that can be pointed at in the event stream. Measurements
are the agent's own evals (`evaluate.py`); `@150` / `@300` / `@1319` is `--limit`,
and `/4000` marks the runs at the evaluator's default 4,000-token cap (everything
else used `--max-tokens 1024`). `official_accuracy` is left null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 111 | 0.25 | sft | base_model | broad_train.jsonl (302,847) | 2e-5 / 1 ep | killed | — | inconclusive | abandon_line |
| exp-02 | 124 | 0.31 | sft | base_model | broad_train.jsonl (302,847) | 2e-5 / 1 ep | killed | — | inconclusive | abandon_line |
| exp-03 | 146 | 0.35 | sft | base_model | broad_train.jsonl (302,847) | 2e-5 / 1 ep | killed | — (paused at step 2,015) | inconclusive | adopt |
| exp-04 | 287 | 1.31 | decode-config | exp-03 | — | — | completed | 0.393 @150 | inconclusive | adopt |
| exp-05 | 298 | 1.34 | sft | exp-03 | broad_train.jsonl (301,991 retained) | 2e-5 / 1 ep (resumed) | completed | 0.440 @150 | supported | adopt |
| exp-06 | 617 | 2.93 | sft | exp-05 | focus_train.jsonl (122,890) | 8e-6 / 1 ep | completed | 0.407 @150 | contradicted | reject |
| exp-07 | 737 | 3.75 | dpo (ORPO) | exp-05 | preferences.jsonl (3,188 pairs) | 5e-7 / 5 ep | completed | 0.380 @150 | contradicted | reject |
| exp-08 | 766 | 4.07 | sft | exp-05 | system_train.jsonl (7,371) | 2e-6 / 1 ep | completed | 0.447 @150 | contradicted | reject |
| exp-09 | 800 | 4.33 | sft | exp-05 | system_train.jsonl (7,371) | 5e-7 / 1 ep | completed | 0.553 @150 | supported | reject |
| exp-10 | 836 | 4.56 | sft | exp-05 | system_train.jsonl (7,371) | 8e-7 / 1 ep | completed | 0.620 @150; 0.6164 @1319; 0.5883 @1319/4000 | supported | adopt |
| exp-11 | 871 | 4.92 | rft | exp-10 | rejection_sft.jsonl (7,371) | 8e-7 / 1 ep | completed | 0.573 @150 | contradicted | reject |
| exp-12 | 972 | 5.44 | dpo | exp-10 | exact_preferences.jsonl (3,616 pairs) | 5e-7 / 1 ep | completed | 0.580 @150 | contradicted | reject |
| exp-13 | 1010 | 5.81 | sft | exp-10 | broad_train.jsonl (1,000 steps) | 2e-6 / 0.21 ep | completed | 0.567 @150 | contradicted | reject |
| exp-14 | 1045 | 6.34 | merge | exp-10 | — (alpha 0.25 toward exp-11) | — | completed | 0.620 @150 | inconclusive | reject |
| exp-15 | 1057 | 6.39 | sft | exp-10 | system_train.jsonl (7,371) | 3e-7 / 1 ep | completed | 0.620 @150; **0.61865 @1319/4000** | supported | adopt |
| exp-16 | 1089 | 6.73 | merge | exp-05 | — (alpha 1.0 packaging of broad step-4,000) | — | completed | 0.420 @150 | supported | adopt |
| exp-17 | 1098 | 6.77 | sft | exp-16 | system_train.jsonl (7,371) | 8e-7 / 1 ep | completed | 0.607 @150 | contradicted | reject |
| exp-18 | 1158 | 7.31 | merge | exp-10 | — (alpha 1.5 toward exp-15) | — | completed | 0.6058 @1319/4000 | contradicted | reject |
| exp-19 | 1174 | 7.39 | merge | exp-15 | — (alpha 0.0 packaging to final_model) | — | completed | 0.633 @150/4000 | inconclusive | adopt |
