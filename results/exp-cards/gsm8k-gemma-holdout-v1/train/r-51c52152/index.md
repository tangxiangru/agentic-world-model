| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 146 | 0.26 | sft | base_model | data_gold (gsm8k train gold x1, 7473) | 2e-5 / 3 | completed | 0.5667 @ n=150 (gold_v1_metrics.json) | inconclusive | adopt |
| exp-02 | 279 | 0.66 | rft | base_model | data_rft1 (gold x2 + 26218 self-sampled, 41156) | 2e-5 / 3 | completed | 0.400 @ n=150 (rft_v1_metrics.json) | contradicted | reject |
| exp-03 | 378 | 1.59 | sft | base_model | data_meta (gold x1 + 50k MetaMathQA GSM) | 2e-5 / 3 | completed | 0.420 @ n=150, epoch 2 (meta_e2_metrics.json) | contradicted | reject |
| exp-04 | 453 | 3.04 | rft | base_model | data_rft1 (gold x2 + self-sampled, 41156) | 1e-5 / 2 | completed | 0.4933 @ n=150, epoch 1 (rftv2_e1_metrics.json) | contradicted | reject |
| exp-05 | 499 | 3.69 | sft | base_model | data_gold (7473) | 2e-5 / 4, NEFTune alpha 5.0 | completed | 0.540 @ n=150, epoch 3 (neft_e3_metrics.json) | inconclusive | reject |
| exp-06 | 533 | 4.03 | other (packaging to final_model) | exp-01 | none | none | completed | 0.510 @ n=300 (cmp_gold_v1.json) | inconclusive | reject |
| exp-07 | 537 | 4.07 | rft | base_model | data_rft1 (gold x2 + self-sampled, 41156) | 2e-5 / 2 | completed | 0.3867 @ n=300, epoch 1 (cmp_rft6_ep1.json) | contradicted | reject |
| exp-08 | 617 | 5.39 | sft | base_model | data_fewshot (gold x3, few-shot system message, 22419) | 2e-5 / 1 | completed | 0.5467 @ n=300 (cmp_fewshot.json) | inconclusive | adopt |
| exp-09 | 648 | 5.80 | sft | base_model | data_fewshot2 (gold x4, k<=6, p=0.85, 29892) | 2e-5 / 1 | completed | 0.5133 @ n=300 (cmp_fewshot2.json) | contradicted | reject |
| exp-10 | 709 | 7.18 | other (packaging to final_model) | exp-08 | none | none | completed | 0.554 @ n=1000 (final_1000.json) | inconclusive | adopt |
| exp-11 | 850 | 8.10 | grpo | exp-10 | gsm8k train prompts, built in-process (no file) | 2e-6 / max_steps 1200 (killed at ~421) | killed | 0.5333 @ n=300 (grpo300.json) | contradicted | abandon_line |
