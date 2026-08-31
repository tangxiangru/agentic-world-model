# r-a9cac75f — reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on one H100 80GB.
The digest carries no event timestamps, so `elapsed_h` is `null` on every card.
Every accuracy below is the agent's own `evaluate.py` run, at the `--limit` shown.
`exp-08` is the state of `/home/ben/task/final_model` at the end of the run.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 54 | null | sft | base_model | train.jsonl 446933 (gsm8k 7473 + metamath-gsm 240000 + orca 199460), `<think>` targets | 2e-5 / 2 | killed | none | inconclusive | abandon_line |
| exp-02 | 90 | null | sft | base_model | train.jsonl 312419 (gsm8k x3 + metamath-gsm + orca 50000), `<think>` targets | 2e-5 / 1 | completed | 0.060 @50 (eval_v1.json) | contradicted | reject |
| exp-03 | 182 | null | sft | base_model | train_v2.jsonl 357257 (gsm8k x9 + metamath-gsm + orca 50000), ANSWER-line targets | 2e-5 / 1 | completed | 0.060 @50 (eval_v2_50.json) | contradicted | adopt |
| exp-04 | 220 | null | decode-config | exp-03 | none (eos_token_id += 151645) | — | completed | 0.433 @150 (eval_v2_fix_150.json); 0.380 @50 | supported | adopt |
| exp-05 | 236 | null | sft | exp-04 | train_v2.jsonl 327365 (gsm8k x5 + metamath-gsm + orca 50000) | 1e-5 / 1 | completed | 0.573 @150 (eval_e2_150.json) | supported | adopt |
| exp-06 | 270 | null | other (packaging) | exp-05 | none (cp final_model_e2 -> final_model) | — | completed | 0.460 @150 (eval_final_150.json) | inconclusive | adopt |
| exp-07 | 277 | null | sft | exp-06 | openai/gsm8k train 7473 only | 5e-6 / 3 | completed | 0.447 @150 (eval_focused_150.json) | contradicted | reject |
| exp-08 | 312 | null | decode-config | exp-06 | none (do_sample=false, temperature=0.0) | — | completed | 0.760 @150 (eval_final_greedy.json); 0.727 / 0.747 / 0.753 on repeats | supported | adopt |
| exp-09 | 326 | null | decode-config | exp-07 | none (do_sample=false, temperature=0.0) | — | completed | 0.740 @150 (eval_focused_greedy.json); 0.733 on repeat | supported | reject |

Comparator for the whole run: base model 0.120 @50 (`baseline_results.json`, event [32]).
Smoke run, not a card: [73], a second foreground `python3 train.py` to inspect the
script's startup output, killed at [80] with the exp-01 launch.
