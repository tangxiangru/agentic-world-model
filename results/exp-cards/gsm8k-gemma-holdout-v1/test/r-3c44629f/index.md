# r-3c44629f — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 16 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 208 | 0.40 | sft (LoRA r=64) | base_model | gsm8k train CoT x2 + MetaMathQA GSM subset, 94,946 | 1e-4 / 1 | failed | none (OOM in the loss at step 100 of 1484) | inconclusive | abandon_line |
| exp-02 | 334 | 0.75 | sft (LoRA r=64, Liger fused CE) | base_model | gsm8k train CoT x2 + MetaMathQA GSM subset, 94,946 | 1e-4 / 1 | completed | accuracy 0.682, n=500 (runs/sft_v1_500.json) | inconclusive | adopt |
| exp-03 | 673 | 1.77 | rft (STaR, LoRA r=64) | exp-02 | self-sampled correct 26,643 + gsm8k-native 7,473 = 34,116 | 1e-4 / 2 | failed | none (save aborted: GenerationConfig invalid) | inconclusive | abandon_line |
| exp-04 | 729 | 2.29 | rft (STaR, LoRA r=64) | exp-02 | self-sampled correct 26,643 + gsm8k-native 7,473 = 34,116 | 1e-4 / 2 | completed | accuracy 0.682, n=500 (runs/sft_rft1_500.json) | contradicted | adopt |
| exp-05 | 800 | 2.94 | grpo (num_gen 8) | exp-04 | gsm8k train prompts, reward = answer correctness | 1e-6 / 150 steps | completed | accuracy 0.68, n=500 (runs/grpo_v1_500.json) | contradicted | reject |
| exp-06 | 836 | 3.03 | other (packaging: sft_v1 copied to final_model) | exp-02 | — | — / — | completed | none (this copy was never evaluated) | inconclusive | adopt |
| exp-07 | 899 | 3.74 | grpo (num_gen 8) | exp-04 | gsm8k train prompts, reward = answer correctness | 6e-6 / 250 steps planned | killed | none (killed at step 49, before checkpoint-50) | inconclusive | abandon_line |
| exp-08 | 994 | 3.99 | grpo (num_gen 8) | exp-04 | gsm8k train prompts, reward = answer correctness | 1e-5 / 250 steps planned | killed | none (killed at step 23) | inconclusive | abandon_line |
| exp-09 | 1063 | 4.15 | sft (full fine-tune of the language model) | base_model | MetaMathQA 40K + gsm8k-native + self-sampled = 74,116 | 1e-5 / 2 | killed | none (killed after ~4 min on a 2.2 h ETA) | inconclusive | abandon_line |
| exp-10 | 1078 | 4.20 | grpo (num_gen 8) | exp-04 | gsm8k train prompts, reward = answer correctness | 3e-5 / 200 steps | completed | accuracy 0.728, n=500; 0.7074, n=1319 | supported | adopt |
| exp-11 | 1249 | 5.09 | other (packaging: grpo_v4 copied to final_model) | exp-10 | — | — / — | completed | accuracy 0.733, n=150; 0.702, n=1000 | inconclusive | adopt |
| exp-12 | 1253 | 5.10 | grpo (num_gen 8, round 2) | exp-10 | gsm8k train prompts, reward = answer correctness | 3e-5 / 200 steps | completed | accuracy 0.716, n=500 (runs/grpo_v5_500.json) | contradicted | reject |
| exp-13 | 1352 | 5.99 | merge (adapter checkpoint-125 into ckpt/sft_rft1) | exp-10 | — | — / — | completed | accuracy 0.707, n=1000 (runs/ck125_1000.json) | inconclusive | reject |
| exp-14 | 1407 | 6.20 | grpo (num_gen 16) | exp-04 | gsm8k train prompts, reward = answer correctness | 3e-5 / 120 steps | completed | accuracy 0.719, n=1000 (runs/grpo_v6_1000.json) | supported | adopt |
| exp-15 | 1516 | 6.98 | other (packaging: grpo_v6 copied to final_model) | exp-14 | — | — / — | completed | accuracy 0.7263, n=1319 (runs/v6_full1319.json) | supported | adopt |
| exp-16 | 1566 | 7.18 | grpo (num_gen 16, continuation) | exp-14 | gsm8k train prompts, reward = answer correctness | 2e-5 / 80 steps | completed | accuracy 0.7165, n=1319 (runs/v7_full1319.json) | contradicted | reject |

Notes

- The submitted artefact is exp-15: `{dir}/final_model` is written three times in this run
  ([836] from ckpt/sft_v1, [1249] from ckpt/grpo_v4, [1516] from ckpt/grpo_v6) and the last
  write is never overwritten; it is verified present, greedy (temperature 0.0) and
  architecturally intact at [2043] and [2062]. Its weights are the exp-14 checkpoint, so
  exp-14 and exp-15 are the same model measured through two paths.
- Smoke and dry runs are not cards. Four pipeline probes before the first SFT ([164],
  [180], [186], [197]) sit on exp-01; the Liger stress test [320] on exp-02; the save-path
  test [723] on exp-04; the 3-step GRPO test [794] on exp-05.
- Two launch commands appear in the stream but never started a process: a GRPO relaunch at
  [981] ("v3 didn't launch", [993], relaunched as exp-08) and one at [1069] (relaunched as
  exp-10). They are recorded in the notes of the cards that did run, not as cards.
- The baseline eval of the base model at [64] is not a card - it produced no candidate. It
  is the comparator on exp-01 and exp-02: 0.08 at --limit 150 with the base model's own
  sampling config [157]. Because it was never re-measured at --limit 500 greedy, the SFT
  card that first produced a real score (exp-02, 0.682) has no like-for-like comparator and
  its verdict is `inconclusive` despite the size of the jump.
- Data generation runs are recorded as `setup.data[].built_by` / `build_command`, not as
  cards: prep_data.py at [126] and [667]/[679], gen_vllm.py (the rejection sampling that
  produced 26,643 verified solutions, 7099/7473 train questions solved) at [603], and
  combine_data.py at [667]/[679].
- The pass@1 vs pass@8 probe at [888] (0.682 vs 0.904 on 500 test questions) is the pivot of
  the run and is recorded as `problem.evidence` and `diagnostic_result` on exp-07, the first
  launch it motivated.
- Workspace coverage is thin: the snapshot holds only SUMMARY.md, results.md, evaluate.py,
  the two judgement files and the timer/monitor logs. src/, data/, ckpt/, logs/ and runs/
  are absent, so every training script, dataset and eval JSON cited in these cards is taken
  from the event stream. No hyper-parameter was filled from an argparse default; where a
  value comes from the script text as written into the stream rather than from the launch
  flags, `hyperparams.other` says so.
- Measured accuracies all come from the agent's own evals; results.md in the snapshot
  independently records 0.080, 0.682, 0.682, 0.68, 0.728, 0.716, 0.707, 0.7074, 0.7263,
  0.713 and 0.7165. The two numbers not in results.md (exp-14's 0.719 at n=1000, exp-15's
  0.693 at n=150) are the agent's statements at [1515] and [1528].
- Eval noise is documented by the run itself: identical weights under the identical default
  invocation scored 0.693 and then 0.713 on 150 items ([1528], [2038]), which is why the
  final model choice was settled on the full 1319-item test set.
