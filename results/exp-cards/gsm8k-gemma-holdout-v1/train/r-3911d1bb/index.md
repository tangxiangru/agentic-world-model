# r-3911d1bb — reconstructed experiment cards (train side)

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100
13 cards, one per launch. No timestamps in this stream, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 70 | null | sft | base_model | gsm8k train, 461 ex (subset 512) | 2e-4 / 1 | failed | none | inconclusive | iterate |
| exp-02 | 79 | null | sft | base_model | gsm8k train, 461 ex (subset 512) | 2e-4 / 1 | completed | none (train_loss 0.504, 29 steps) | inconclusive | adopt |
| exp-03 | 85 | null | merge | exp-02 | none | null / null | completed | none (eval60 ran, score never printed) | inconclusive | reject |
| exp-04 | 106 | null | sft | base_model | gsm8k train, 922 ex (subset 1024) | 5e-5 / 1 | completed | none | inconclusive | adopt |
| exp-05 | 111 | null | merge | exp-04 | none | null / null | completed | accuracy 0.35 (n=60) | inconclusive | reject |
| exp-06 | 120 | null | sft | base_model | gsm8k train, 1844 ex (subset 2048) | 5e-5 / 2 | completed | none | inconclusive | adopt |
| exp-07 | 127 | null | merge | exp-06 | none | null / null | completed | none (eval60 ran, score never printed) | inconclusive | reject |
| exp-08 | 138 | null | sft | base_model | gsm8k train, 1844 ex (subset 2048) | 5e-5 / 1 | completed | none | inconclusive | adopt |
| exp-09 | 145 | null | merge | exp-08 | none | null / null | completed | none (eval60 ran, score never printed) | inconclusive | reject |
| exp-10 | 152 | null | sft | base_model | gsm8k train, 1844 ex (subset 2048) | 2e-5 / 1 | completed | none (train_loss 0.487, 116 steps) | inconclusive | adopt |
| exp-11 | 157 | null | merge | exp-10 | none | null / null | completed | accuracy 0.35 (n=60) | inconclusive | reject |
| exp-12 | 170 | null | sft | base_model | gsm8k train, 7217 ex (full split) | 2e-5 / 1 | completed | none | inconclusive | adopt |
| exp-13 | 185 | null | merge | exp-12 | none | null / null | completed | accuracy 0.34 (n=150), final_eval_150.json | inconclusive | adopt |

Submitted artifact: exp-13 — `merge_lora.py --adapter-path runs/final-lora --output-path final_model` at [185], scored 0.34 on 150 samples at [194]/[197]. Nothing in the stream overwrites `final_model` afterwards.
