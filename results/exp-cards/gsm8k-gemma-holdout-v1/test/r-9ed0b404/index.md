# r-9ed0b404 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 6 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 64 | 0.25 | sft | base_model | gsm8k main (train), 7473 | 2e-4 / 3 | killed | none (cut off at step 53/702 by the 120 s tool timeout) | inconclusive | abandon_line |
| exp-02 | 82 | 0.32 | sft | base_model | gsm8k main (train), 7473 | 2e-4 / 3 | completed | none (train_loss 0.7292 @702 steps, train.log) | inconclusive | adopt |
| exp-03 | 114 | 0.40 | merge (sft_output/checkpoint-100 → sft_merged_100) | exp-02 | — | — / — | completed | none (limit-10 eval crashed: vLLM server would not start) | inconclusive | abandon_line |
| exp-04 | 162 | 5.55 | merge (sft_final → sft_merged) | exp-02 | — | — / — | completed | none (never evaluated) | inconclusive | adopt |
| exp-05 | 294 | 5.75 | rft | exp-04 | synthetic:self, gsm8k_rft_train.parquet, 17639 | 1e-4 / 2 | completed | none (train_loss 0.3957 @1104 steps, rft_train.log) | inconclusive | adopt |
| exp-06 | 350 | 5.81 | merge (packaging: rft_final onto sft_merged → final_model + full-test eval) | exp-05 | — | — / — | completed | accuracy 0.4875, n=1319 (evaluation_results.json) | inconclusive | adopt |

Notes

- The submitted artifact is **exp-06** (`final_model` in the agent's workspace), the only candidate the agent
  left in place and the only one ever scored: 0.4875 accuracy, stderr 0.0138, on all 1319
  gsm8k test samples [436][437][471][472], `evaluation_results.json`.
- Every verdict is `inconclusive`: the run's only comparator is the base-model eval at
  `--limit 5` (accuracy 0.000) at [23], and no SFT-stage checkpoint was ever scored, so the
  final number has nothing to be measured against under the same protocol. The agent's
  "massive improvement over the baseline" [473] compares a 1319-sample score to a 5-sample one.
- The agent stated no problem and no hypothesis before any launch: the digest's only two
  `say` events, [452] and [473], both come after the final eval. `stated_by_agent` is
  `false` on all six cards, and the only reasoning in the agent's own words lives in the
  code comments of `train_rft.py` ("Lower LR for second stage", "2 is safe").
- The pipeline is one straight line: SFT LoRA on raw gsm8k (exp-01 killed → exp-02) → merge
  (exp-04, with the step-100 dead end exp-03 alongside) → sample 4 solutions/question with
  vLLM and keep the 17639 of 29892 whose `#### <n>` matches gold → RFT LoRA (exp-05) → merge
  onto the merged SFT weights and evaluate (exp-06).
- Smoke runs and API-fighting launches are not cards. Recorded as `provenance.smoke_runs`:
  the six trainer-API crashes at [30][34][38][50][54][58] (exp-01), the argument-less merge
  at [110] (exp-03), the pipeline whose `train_rft.py` died on `fix_mistral_regex=True`
  at [198] (exp-05), and the first `finish.sh` at [320], killed at [348] while still in its
  wait loop so the merge base could be corrected (exp-06). The line drawn: a launch that
  performed some of the work it existed to do and was then killed is a card (exp-01, 53
  training steps); one that crashed on a library/argument error, or did nothing before being
  stopped and relaunched with a fixed argument, is a smoke run.
- The three `generate_rft.py` launches ([166] crashed in vLLM startup, [172] cut off by the
  tool timeout, [174] completed under nohup) build training data rather than a candidate, so
  they appear as `setup.data[].build_command` on exp-05, not as cards.
- `elapsed_h` is the `t=+H.HHh` of each launch block. The stream has a 4.6 h gap between
  [157] (t=+0.92h) and [158] (t=+5.55h) that no event explains; the SFT run it was
  nominally waiting on took 0.35 h.
