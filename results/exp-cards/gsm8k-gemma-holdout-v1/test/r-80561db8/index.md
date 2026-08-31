# r-80561db8 - reconstructed experiment cards

12 cards. Base model google/gemma-3-4b-pt, benchmark gsm8k, 10 h budget, one H100. The incumbent from t=+4.9h onward is exp-02; exp-12 is the submitted final_model (exp-02's weights plus a greedy generation_config). Smoke tests are not cards: [63] is on exp-01, [172] and [175] on exp-03.

The workspace snapshot holds only the agent's scripts (no `logs/`, no `eval_*.json`, no `data/`), so every `path` in `result.measurements`, `problem.evidence` and `setup.data` is the run's own absolute path as it appears in the stream, not a file present in the snapshot; the scripts each card cites are listed in its `provenance.snapshot_files`. No base-model baseline was ever measured, so exp-01 has no comparator.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 76 | 0.12 | sft | base_model | sft_gsm.jsonl | 2e-5 / 2 | completed | 0.387 @150 | inconclusive | adopt |
| exp-02 | 213 | 2.51 | sft | exp-01 | sft_fewshot.jsonl | 1e-5 / 1 | completed | 0.573 @150 | supported | adopt |
| exp-03 | 255 | 4.93 | grpo | exp-02 | gsm8k train + MetaMath GSM_* prompts (10k) | 5e-6 / 500 steps | completed | 0.507 @150 | contradicted | reject |
| exp-04 | 334 | 6.78 | rft | exp-02 | rft_mixed.jsonl | 8e-6 / 2 | killed | - | inconclusive | abandon_line |
| exp-05 | 347 | 6.83 | rft | exp-02 | rft_mixed.jsonl | 8e-6 / 1 | killed | - | inconclusive | abandon_line |
| exp-06 | 364 | 6.85 | rft | exp-02 | rft_mixed.jsonl | 8e-6 / 1 | killed | - | inconclusive | abandon_line |
| exp-07 | 372 | 6.86 | rft | exp-02 | rft_mixed.jsonl | 8e-6 / 1 | failed (CUDA OOM) | - | inconclusive | abandon_line |
| exp-08 | 378 | 6.87 | rft | exp-02 | rft_mixed.jsonl | 8e-6 / 1 | killed | - | inconclusive | abandon_line |
| exp-09 | 390 | 6.88 | rft | exp-02 | rft_train.jsonl (missing at launch) | 1e-5 / 1 | failed (no data file) | - | inconclusive | abandon_line |
| exp-10 | 399 | 6.89 | rft | exp-02 | rft_train.jsonl | 1e-5 / 1 | completed | 0.507 @150 | contradicted | reject |
| exp-11 | 450 | 8.57 | sft | exp-02 | sft_gsm_fs10.jsonl | 5e-6 / 900 steps | completed | 0.527 @150 | contradicted | reject |
| exp-12 | 493 | 9.58 | decode-config | exp-02 | - | - / - | completed | 0.600 @150 | inconclusive | adopt |
