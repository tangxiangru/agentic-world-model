# r-9f1c9470 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 90 | null | sft (LoRA r=64) | base_model | openai/gsm8k train (7,473) | 2e-4 / 4.0 | completed | none (adapter never scored; merged as exp-02) | inconclusive | adopt |
| exp-02 | 127 | null | merge | exp-01 | none | null / null | completed | 0.175 acc, n=80 official slice (experiments/exp1_eval80.json); re-eval after the stop-token change 0.175 (experiments/exp1_eval80_stopfix.json) | inconclusive | adopt |
| exp-03 | 161 | null | sft (full fine-tune) | base_model | openai/gsm8k train (7,473) | 2e-5 / 2.0 | completed | 0.025 acc, n=80 official slice (experiments/exp2_eval80.json), -0.15 vs exp-02 | contradicted | reject |
| exp-04 | 235 | null | sft (LoRA r=64, dropout 0) | base_model | openai/gsm8k train (7,473) | 5e-5 / 2.0 | completed | none (adapter never scored; its merge's eval was killed) | inconclusive | adopt |
| exp-05 | 248 | null | merge | exp-04 | none | null / null | completed | none (eval at [252] killed at [265] before any accuracy) | inconclusive | abandon_line |
| exp-06 | 291 | null | other (packaging: cp to final_model) | exp-02 | none | null / null | completed | 0.187 acc, n=150 official slice (final_eval150.json) | inconclusive | adopt |
