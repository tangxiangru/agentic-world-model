| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 50 | null | sft | base_model | train_data.jsonl | 0.0002 / 2 | failed | - | inconclusive | abandon_line |
| exp-02 | 63 | null | sft | base_model | train_data.jsonl | 0.0002 / 2 | killed | - | inconclusive | abandon_line |
| exp-03 | 82 | null | sft | base_model | train_data.jsonl | 0.0002 / 1 | completed | 0.340 @50 | inconclusive | reject |
| exp-04 | 161 | null | sft | base_model | train_data_v2.jsonl | 0.0002 / 2 | completed | 0.280 @50 | contradicted | reject |
| exp-05 | 199 | null | sft | base_model | train_data_v2.jsonl | 2e-05 / 3 | completed | 0.280 @50 | contradicted | adopt |
| exp-06 | 261 | null | decode-config | exp-05 | - | - / - | completed | 0.500 @50 | supported | adopt |
| exp-07 | 275 | null | sft | base_model | train_data_v2.jsonl | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-08 | 344 | null | sft | base_model | train_data_v3b.jsonl | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-09 | 359 | null | sft | base_model | train_data_v3b.jsonl | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-10 | 375 | null | sft | base_model | train_data_v3b.jsonl | 2e-05 / 3 | killed | - | inconclusive | abandon_line |
| exp-11 | 380 | null | sft | exp-05 | train_data_v3b.jsonl | 5e-06 / 1 | completed | 0.760 @50 | supported | adopt |
| exp-12 | 508 | null | sft | exp-11 | train_data_v4.jsonl | 5e-06 / 1 | killed | - | inconclusive | abandon_line |
| exp-13 | 511 | null | other | exp-11 | - | - / - | completed | - | inconclusive | adopt |
| exp-14 | 530 | null | sft | exp-11 | train_data_v4.jsonl | 5e-06 / 1 | killed | - | inconclusive | abandon_line |
| exp-15 | 552 | null | sft | exp-11 | train_data_v3b.jsonl | 2e-06 / 1 | completed | 0.700 @50 | contradicted | reject |
| exp-16 | 596 | null | sft | base_model | train_data_v3b.jsonl | 2e-05 / 2 | killed | - | inconclusive | abandon_line |
| exp-17 | 611 | null | decode-config | exp-13 | - | - / - | failed | - | inconclusive | abandon_line |
| exp-18 | 654 | null | decode-config | exp-13 | - | - / - | completed | 0.760 @50 | inconclusive | reject |
| exp-19 | 666 | null | decode-config | exp-13 | - | - / - | completed | 0.840 @50 | supported | reject |
| exp-20 | 676 | null | decode-config | exp-13 | - | - / - | completed | 0.813 @150 | supported | iterate |
| exp-21 | 681 | null | decode-config | exp-13 | - | - / - | completed | 0.787 @150 | supported | reject |
| exp-22 | 686 | null | decode-config | exp-13 | - | - / - | completed | 0.800 @150 | supported | reject |
| exp-23 | 691 | null | decode-config | exp-13 | - | - / - | completed | 0.787 @150 | supported | reject |
| exp-24 | 696 | null | decode-config | exp-13 | - | - / - | completed | 0.793 @150 | supported | reject |
| exp-25 | 701 | null | decode-config | exp-13 | - | - / - | completed | 0.800 @150 | supported | adopt |
| exp-26 | 708 | null | sft | base_model | train_data_v3b.jsonl | 2e-05 / 2 | killed | - | inconclusive | abandon_line |
