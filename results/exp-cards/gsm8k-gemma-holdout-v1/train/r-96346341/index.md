| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 418 | 0.18 | sft | base_model | data/sft.jsonl (omi2-gsm 65k / omi2-math 15k / orca 15k caps, fewshot 0.35; size not printed) | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-02 | 713 | 0.35 | sft | base_model | data/sft.jsonl, 57,380 loaded (omi2-gsm 42k / omi2-math 8k / orca 8k caps, fewshot 0.25) | 1e-5 / 1 | completed | accuracy 0.887 @150 (+0.534 vs base 0.353), logs/sft1_greedy.json | supported | adopt |
| exp-03 | 950 | 1.60 | grpo | exp-02 | openai/gsm8k train prompts (fewshot 0.3, k up to 10) | 1e-6 / 1.0 | failed | none (OOM) | inconclusive | abandon_line |
| exp-04 | 1069 | 6.68 | grpo | exp-02 | openai/gsm8k train prompts (fewshot 0.15, k 1-4) | 1e-6 / 1.0 | failed | none (crash in save_pretrained) | inconclusive | abandon_line |
| exp-05 | 1086 | 6.68 | other (package cp -> final_model) | exp-02 | none | n/a | completed | none (500-sample eval killed at timer expiry) | inconclusive | reject |
| exp-06 | 1171 | 7.90 | grpo | exp-02 (via runs/grpo_init) | openai/gsm8k train prompts (fewshot 0.15, k 1-4), 40 steps | 1e-6 / 1.0 | killed | accuracy 0.912 @500 (logs/grpo40_500.json); 0.900 @150 (+0.013 vs exp-02), logs/grpo40.json | supported | adopt |
| exp-07 | 1270 | 10.04 | other (package cp + bf16 re-save -> final_model) | exp-06 | none | n/a | completed | none (verification eval at [1324] has no result in the stream) | inconclusive | adopt |
