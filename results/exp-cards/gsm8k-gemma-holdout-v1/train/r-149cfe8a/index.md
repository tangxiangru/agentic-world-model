# Extracted experiment cards

Base model post-trained: HuggingFaceTB/SmolLM3-3B-Base | benchmark: gsm8k | budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 208 | 0.30 | sft | base_model | data/train.jsonl (37,473: gsm8k-train + MetaMath GSM_AnsAug/GSM_Rephrased) + data/fewshot_pool.jsonl (7,473) | 1e-5 / 3 | completed | 0.6667 @150 (results/sft1_final.json) | inconclusive | adopt |
| exp-02 | 374 | 1.58 | decode-config | exp-01 | - | - / - | completed | 0.800 @150 (results/sft1_ep2.json); 0.787 @150 (results/sft1_final_greedy.json) | supported | adopt |
| exp-03 | 416 | 1.67 | other (packaging) | exp-01 | - | - / - | completed | 0.800 @150 (results/sft1_ep2.json, copied checkpoint) | inconclusive | adopt |
| exp-04 | 443 | 1.80 | rft | base_model | data/train_rft.jsonl = data/train.jsonl + data/rft.jsonl (26,806 self-samples, 7228/7473 solved) | 1e-5 / 3 | completed | 0.827 @150 (results/sft2_ep3.json); 0.801 @1319 (results/sft2_ep3_full.json) | supported | adopt |
| exp-05 | 511 | 4.00 | other (packaging) | exp-04 | - | - / - | completed | 0.801 @1319 (results/sft2_ep3_full.json, copied checkpoint) | inconclusive | adopt |
| exp-06 | 568 | 4.16 | grpo | exp-04 | openai/gsm8k[train] prompts, 2-shot | 1e-6 / 500 steps | failed | none (crashed at the step-100 save) | inconclusive | abandon_line |
| exp-07 | 627 | 4.43 | grpo | exp-04 | openai/gsm8k[train] prompts, 2-shot | 2e-6 / 500 steps | completed | 0.818 @1319 (results/grpo_500.json) | supported | adopt |
| exp-08 | 682 | 5.61 | other (packaging) | exp-07 | - | - / - | completed | 0.818 @1319 (results/grpo_500.json, copied checkpoint) | inconclusive | adopt |
| exp-09 | 682 | 5.61 | grpo | exp-07 | openai/gsm8k[train] prompts, 2-shot | 2e-6 / 600 steps | completed | 0.8203 @1319 (results/grpo2_400.json) | inconclusive | adopt |
| exp-10 | 740 | 6.92 | other (packaging) | exp-09 | - | - / - | completed | 0.8733 @150 default settings (results/final_default150.json); 0.8180 @1319 (results/FINAL_full.json) | inconclusive | adopt |
| exp-11 | 830 | 7.33 | grpo | exp-04 | openai/gsm8k[train] prompts, 4-shot | 2e-6 / 400 steps | completed | 0.8533 @150 (peak, path not stated); 0.840 @150 (results/grpo3_f150_400.json) | contradicted | reject |

Submitted artifact: exp-10 (final_model = runs/grpo2/checkpoint-400).
