| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 1686 | 5.41 | decode-config | base_model | - | - | killed | - | inconclusive | abandon_line |
| exp-02 | 1719 | 5.54 | decode-config | base_model | - | - | completed | 0.74 @50 | contradicted | reject |
| exp-03 | 1800 | 5.83 | rft | base_model | rft_current_exact_disablethinking_greedy_500.jsonl (n=502) | 5e-07 / - | completed | 0.72 @50 | contradicted | reject |
| exp-04 | 1859 | 5.96 | decode-config | exp-03 | - | - | completed | 0.76 @50 | supported | reject |
| exp-05 | 1867 | 6.0 | merge | base_model | - | - | completed | 0.68 @50 | contradicted | reject |
| exp-06 | 1925 | 6.08 | decode-config | base_model | - | - | completed | - | inconclusive | reject |
| exp-07 | 1998 | 6.4 | dpo | base_model | dpo_current_exact_wrong_160.jsonl (n=157) | 5e-07 / - | failed | - | inconclusive | abandon_line |
| exp-08 | 2006 | 6.42 | dpo | base_model | dpo_current_exact_wrong_160.jsonl (n=157) | 5e-07 / - | completed | 0.76 @400 | supported | adopt |
| exp-09 | 2043 | 6.94 | dpo | base_model | dpo_current_exact_wrong_320.jsonl (n=320) | 4e-07 / - | completed | 0.796875 @64 (dev proxy) | contradicted | reject |
| exp-10 | 2061 | 7.09 | other | exp-08 | - | - | completed | 0.76 @400 | inconclusive | adopt |
| exp-11 | 2081 | 7.12 | merge | exp-10 | - | - | completed | 0.75 @100 | contradicted | reject |
| exp-12 | 2087 | 7.16 | decode-config | exp-10 | - | - | completed | 0.77 @100 | contradicted | reject |
| exp-13 | 2187 | 7.53 | dpo | exp-10 | dpo_current_exact_wrong_cleaned_220.jsonl (n=201) | 4e-07 / - | completed | 0.7 @50 | contradicted | reject |
| exp-14 | 2229 | 7.65 | sft | exp-10 | sft_current_exact_wrong_cleaned_220.jsonl (n=201) | 8e-07 / 2.0 | completed | 0.68 @50 | contradicted | reject |
| exp-15 | 2261 | 7.76 | sft | exp-10 | sft_exact_mix_correct201_plus_gold201.jsonl (n=402) | 4e-07 / 1.0 | completed | 0.8125 @64 (dev proxy) | contradicted | reject |
| exp-16 | 2301 | 7.84 | decode-config | exp-10 | - | - | completed | 0.22 @50 | contradicted | reject |
| exp-17 | 2320 | 7.94 | decode-config | exp-10 | - | - | completed | 0.752 @400 | contradicted | reject |
| exp-18 | 2351 | 8.03 | merge | exp-10 | - | - | completed | 0.76 @100 | contradicted | reject |
| exp-19 | 2407 | 8.35 | dpo | exp-10 | dpo_self_current_exact_sampled_160_semiclean.jsonl (n=96) | 4e-07 / - | completed | 0.18 @50 | contradicted | reject |
| exp-20 | 2458 | 8.5 | decode-config | exp-10 | - | - | completed | 0.76 @50 | contradicted | reject |
| exp-21 | 2515 | 8.58 | dpo | exp-10 | dpo_current_exact_wrong_cleaned_220.jsonl (n=201) | 2e-07 / - | completed | 0.16 @50 | contradicted | reject |
| exp-22 | 2605 | 8.76 | decode-config | exp-10 | - | - | completed | 0.752 @400 | contradicted | reject |
