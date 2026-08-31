# r-635b683e — reconstructed experiment cards

Base model Qwen/Qwen3-4B-Base, benchmark gsm8k, 10 h on one H100.
Comparator for the whole run: the untouched base model at 0.465, n=200, logs/baseline.json [242].
All accuracies are the agent's own evals; the run's official score is not part of these cards.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 447 | 0.19 | sft | base_model | sft_big.jsonl (95136) | 1e-5 / 1 epoch | killed | — | inconclusive | abandon_line |
| exp-02 | 591 | 0.30 | sft | base_model | sft_big.jsonl (95136) | 1e-5 / 1 epoch | completed | 0.930 at n=200; 0.920 at n=300 | supported | adopt |
| exp-03 | 747 | 1.60 | decode-config | exp-02 | — | — | completed | 0.855 at n=200 (vs 0.930 greedy) | supported | reject |
| exp-04 | 810 | 1.64 | other (packaging to final_model) | exp-02 | — | — | completed | — | inconclusive | adopt |
| exp-05 | 847 | 1.91 | grpo | exp-02 | hard.jsonl | 1e-6 / 260 steps planned | failed | — | inconclusive | abandon_line |
| exp-06 | 863 | 1.96 | grpo | exp-02 | hard.jsonl | 1e-6 / 260 steps planned | failed | — | inconclusive | abandon_line |
| exp-07 | 888 | 2.02 | grpo | exp-02 | hard.jsonl | 1e-6 / 260 steps planned | killed | — | inconclusive | abandon_line |
| exp-08 | 894 | 2.03 | grpo | exp-02 | hard.jsonl | 1e-6 / 14 of 260 steps | failed | — | inconclusive | abandon_line |
| exp-09 | 978 | 2.51 | grpo | exp-02 | hard.jsonl | 1e-6 / 8 of 240 steps | failed | — | inconclusive | abandon_line |
| exp-10 | 1012 | 2.58 | grpo | exp-02 | hard.jsonl | 1e-6 / 240 steps | completed | 0.937 at n=300 (step 150) | inconclusive | adopt |
| exp-11 | 1168 | 4.34 | grpo | exp-10 | hard.jsonl | 3e-6 / 50 of 180 steps | failed | — | inconclusive | abandon_line |
| exp-12 | 1236 | 4.72 | grpo | exp-10 | hard.jsonl | 3e-6 / 160 steps | failed | 0.930 at n=300 (step 120); 0.877 (step 80) | inconclusive | adopt |
| exp-13 | 1308 | 6.05 | merge | exp-02 + exp-10 + exp-12 | — | — | completed | 0.930 at n=300 | inconclusive | adopt |
| exp-14 | 1373 | 7.61 | other (packaging to final_model) | exp-13 | — | — | completed | 0.927 at n=150 | inconclusive | adopt |

Submitted model: exp-14 — final_model is a copy of the exp-13 weight soup (equal average of the
SFT checkpoint, the round-one RL step-150 checkpoint and the round-two step-120 checkpoint),
served greedy. exp-04 held final_model earlier in the run and was overwritten at [1373].
