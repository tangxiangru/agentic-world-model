# r-5cf89e69 — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h on one H100.
No timestamps in this digest, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 77 | null | sft | base_model | train_data.jsonl (247473) | 2e-5 / 2 | failed | — | inconclusive | abandon_line |
| exp-02 | 80 | null | sft | base_model | train_data.jsonl (247473) | 2e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-03 | 92 | null | sft | base_model | train_data.jsonl (247473) | 2e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-04 | 111 | null | sft | base_model | train_data.jsonl (247473) | 2e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-05 | 141 | null | sft | base_model | train_data.jsonl (247473) | 2e-5 / 2 | completed | accuracy 0.400 (n=50, eval_v1_50.json) | inconclusive | adopt |
| exp-06 | 230 | null | sft | base_model | train_data_v3.jsonl (54946) | 1e-5 / 3 | killed | — | inconclusive | abandon_line |
| exp-07 | 248 | null | sft | base_model | train_data_v4.jsonl (22473) | 2e-5 / 3 | completed | accuracy 0.613 (n=150, eval_v2_150.json) | inconclusive | adopt |
| exp-08 | 259 | null | other (package) | exp-05 | — | — / — | completed | — (eval blocked: vLLM could not start) | inconclusive | reject |
| exp-09 | 297 | null | sft | exp-05 | train_data_v4.jsonl (22473) | 5e-6 / 3 | completed | accuracy 0.607 (n=150, eval_v3_150.json) | inconclusive | reject |
| exp-10 | 302 | null | other (package) | exp-07 | — | — / — | completed | accuracy 0.613 (n=150, eval_v2_150.json, measured on the copied weights) | inconclusive | adopt |
| exp-11 | 371 | null | sft | base_model | train_data_v5.jsonl (7463) | 2e-5 / 5 | completed | accuracy 0.500 (n=150, eval_v4_150.json) | contradicted | reject |
| exp-12 | 416 | null | sft | base_model | train_data_v6.jsonl (57473) | 2e-5 / 2 | killed | — | inconclusive | abandon_line |

Submission: exp-10 — the exp-07 weights copied to `final_model` at [302], the last write
to that path anywhere in the stream.
