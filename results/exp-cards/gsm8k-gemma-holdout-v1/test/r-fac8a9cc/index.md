# r-fac8a9cc — reconstructed experiment cards

- base model: google/gemma-3-4b-pt
- benchmark: gsm8k, 10 h budget, 1x H100
- cards: 25 (launches [118] .. [1488]); the digest carries timestamps, so every `elapsed_h` is real
- no smoke runs: every crashed or killed launch here was a full-size run the agent meant as real

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 118 | 0.32 | sft | base_model | data_sft: gsm8k train x4 + OpenMathInstruct-2 (100k) + Orca-Math (15k), 49.3M packed tokens | 2e-4 / 1 | completed | 0.6533 @150 (eval_run1_150.json); 0.6194 @1319; 0.46/0.48 @50 at steps 2k/4k | inconclusive | adopt |
| exp-02 | 438 | 2.05 | sft | exp-01 | data_sft_unique: 72,939 unique OpenMath problems + gsm8k train x2, ~29.6M tokens | 1e-4 / 1 | completed | 0.6267 @150 (eval_run2_150.json, -0.0267 vs exp-01); 0.6247 @1319 | contradicted | reject |
| exp-03 | 686 | 3.24 | grpo | exp-01 | gsm8k train prompts (data_sft/train.jsonl) | 5e-6 / 200 steps planned, 5 run | killed | — (no checkpoint; TRL masked every 106-terminated completion) | inconclusive | abandon_line |
| exp-04 | 701 | 3.28 | grpo | exp-01 | gsm8k train prompts (data_sft/train.jsonl) | 5e-6 / 200 steps | completed | 0.6733 @150 at step 100 (eval_grpo1_step100_150.json, +0.0200 vs exp-01); 0.6255 @1319 | supported | adopt |
| exp-05 | 786 | 4.02 | rft | exp-04 | data_rft1: 19,843 verified self-traces + one human solution per problem (27,304 ex, 7.85M tokens) | 5e-5 / 1 | completed | 0.6368 @1319 (eval_rft1_all_1319.json, +0.0114 vs exp-04); 0.6353 packaged repeat | supported | adopt |
| exp-06 | 829 | 4.42 | sft | base_model | data_metamath: 103,834 MetaMathQA GSM transforms + gsm8k train x3 (126,247 ex, 43.27M tokens) | 2e-4 / 1 | completed | 0.60 @50 at step 3,000 (eval_meta_step3000_50.json); 0.52 final | inconclusive | reject |
| exp-07 | 999 | 6.19 | grpo | exp-04 | data_grpo_curriculum: 1,150 gsm8k train problems with 1-2 verifier-accepted traces | 2e-6 / 100 steps | completed | 0.6209 @1319 at step 50 (eval_curr1_step50_all_1319.json, -0.0045 vs exp-04) | contradicted | reject |
| exp-08 | 1031 | 6.44 | grpo | exp-01 | gsm8k train prompts (data_sft/train.jsonl) | 5e-6 / 100 steps | completed | 0.6270 @1319 (eval_seed2_all_1319.json); 0.62 @50 | inconclusive | reject |
| exp-09 | 1063 | 6.68 | merge | exp-04 | none (GRPO adapter scaled 0.75x) | n/a | completed | 0.60 @50 (eval_scale075_50.json, -0.0400) | contradicted | reject |
| exp-10 | 1066 | 6.69 | merge | exp-04 | none (GRPO adapter scaled 1.25x) | n/a | completed | 0.6171 @1319 (eval_scale125_all_1319.json, -0.0083); 0.66 @150; 0.64 @50 | contradicted | reject |
| exp-11 | 1088 | 6.84 | merge | exp-04 | none (GRPO adapter scaled 1.10x) | n/a | completed | 0.58 @50 (eval_scale110_50.json, -0.0600) | contradicted | reject |
| exp-12 | 1095 | 6.89 | decode-config | exp-04 | none (exact-greedy export of the same weights) | n/a | completed | 0.66 @150 (eval_best_greedy_150.json, -0.0133); 0.6202 @1319 | contradicted | reject |
| exp-13 | 1105 | 6.94 | grpo | exp-01 | gsm8k train prompts (data_sft/train.jsonl) | 5e-6 / 200 steps, schedule-matched, new seed | completed | 0.60 @50 final (eval_seed3_final_50.json, -0.0400); 0.58 @50 at step 100 | contradicted | reject |
| exp-14 | 1149 | 7.30 | sft | exp-04 | data_gsm_only: 7,471 clean gsm8k train solutions x4 (29,884 ex, 8.59M tokens) | 1e-5 / 1 | completed | 0.58 @50 at step 250 (eval_gsm_polish_step250_50.json, -0.0600); 0.54 final | contradicted | reject |
| exp-15 | 1205 | 7.74 | other (packaging to final_model) | exp-04 | none | n/a | completed | 0.6667 @150 (final_eval_150.json, -0.0067); 0.6255 @1319 (final_eval_all_1319.json) | supported | adopt |
| exp-16 | 1314 | 8.58 | decode-config | exp-05 | none (temperature-0.10 and exact-greedy copies of the RFT export) | n/a | completed | 0.6368 @1319 at temp 0.10 (eval_rft_temp010_all_1319.json, +0.0000); 0.6308 greedy | contradicted | reject |
| exp-17 | 1337 | 8.70 | other (packaging to final_model) | exp-05 | none | n/a | completed | 0.6353 @1319 (final_eval_verified_1319.json, +0.0099 vs exp-15) | supported | adopt |
| exp-18 | 1373 | 8.80 | merge | exp-05 | none (RFT adapter scaled 1.15x) | n/a | completed | 0.6315 @1319 (eval_rft_scale115_all_1319.json, -0.0053) | contradicted | reject |
| exp-19 | 1404 | 8.91 | sft | exp-17 | data_rft1 (second pass over the same corpus) | 1e-5 / 300 steps | failed | — (liger_kernel import: source code string cannot contain null bytes) | inconclusive | abandon_line |
| exp-20 | 1408 | 8.92 | sft | exp-17 | data_rft1 (second pass over the same corpus) | 1e-5 / 300 steps | failed | — (same import error with an isolated bytecode cache) | inconclusive | abandon_line |
| exp-21 | 1429 | 8.94 | sft | exp-17 | data_rft1 (second pass over the same corpus) | 1e-5 / 300 steps | completed | 0.6331 @1319 (eval_rft_continue_all_1319.json, -0.0038) | contradicted | reject |
| exp-22 | 1458 | 9.13 | merge | exp-05 | none (RFT adapter scaled 0.90x) | n/a | completed | 0.6452 @1319 (eval_rft_scale090_all_1319.json, +0.0083) but 0.6293 and 0.6315 on two repeats | contradicted | reject |
| exp-23 | 1466 | 9.20 | other (packaging to final_model) | exp-22 | none | n/a | completed | 0.6293 @1319 (final_eval_scale090_verified_1319.json, -0.0159 vs the selection run) | contradicted | reject |
| exp-24 | 1477 | 9.27 | decode-config | exp-22 | none (exact-greedy export of the 0.90x weights) | n/a | completed | 0.6277 @1319 (eval_rft_scale090_greedy_all_1319.json, -0.0174) | contradicted | reject |
| exp-25 | 1488 | 9.39 | other (restore final_model) | exp-17 | none | n/a | completed | 0.6368 / 0.6353 @1319 carried from exp-05 and exp-17 (no new run) | inconclusive | adopt |

Notes

- **Submitted card: exp-17** — `final_model` holds the merged accepted-trace RFT
  export (`run_rft1/model`, the weights of exp-05), packaged and verified at
  838/1,319 = 63.53% from that exact path, against a selection run of
  840/1,319 = 63.68%. It was moved aside at [1466] while the 0.90x
  interpolation was on trial and moved back by exp-25, which is the last launch
  to touch the artifact; the shard hashes were re-checked after the restore
  ([1501]) and match.
- The chain that produced it: exp-01 (decontaminated math SFT, 65.3% @150) ->
  exp-04 step 100 (answer-verified GRPO, 67.3% @150 / 62.55% @1319) -> exp-05
  (SFT on 19,843 verifier-accepted self-generated traces, 63.68% @1319).
- `adopt` on exp-01, exp-04, exp-05 and exp-15 records that the output became the
  incumbent, the packaged `final_model` of the moment, or the parent of a later
  card — not that it was submitted.
- **The 50-item screen mis-ranked the winner.** exp-05 was rejected at [828] on a
  50-problem screen (60% / 62% against the parent's 64%) and only recovered when
  the agent swept the full 1,319-item set at [1297]. exp-06, exp-13 and exp-14
  were closed on that same screen and never re-measured, and exp-08 (independent
  GRPO seed) was declared a failed replication at 62% on 50 items while later
  scoring 62.70% on the full set — slightly *above* the seed-1 checkpoint it was
  supposed to replicate. The agent never revisited the "seed 1 is exceptional"
  conclusion after that measurement.
- Fourteen of the twenty-five verdicts are `contradicted` and only four are
  `supported`; from exp-04 onward every candidate sits between 61.7% and 64.5% on
  the full test set with a per-run standard error of about 1.3 points, so even the
  supported deltas (+2.0 pts @150 for exp-04, +1.1 pts @1319 for exp-05) are
  inside noise. The two remaining `supported` verdicts, exp-15 and exp-17, are
  packaging reproductions rather than method results. The agent itself reached this reading for exp-22, whose 64.52%
  outlier it declined to select after three replicates averaged 63.53%.
- `merge_adapter.py` and `finalize_assets.py` invocations at scale 1.0 that merely
  materialise a training run's own adapter are recorded on that training card
  rather than as separate cards: the same export path is applied identically to
  every checkpoint in the run. Only the deliberate weight-space rescalings
  (0.75x, 1.10x, 1.15x, 1.25x, 0.90x) and the `--exact-greedy` re-exports are
  their own cards.
- Run-level gaps in the digest:
  - The stream opens at [118], which is the first SFT launch itself. The
    `prepare_data.py` run that built `data_sft`, the baseline evaluation that
    produced `baseline20.json` (10% on 20 items), and all pre-launch reasoning
    are outside it, so exp-01's problem and hypothesis are the agent's words at
    [219], narrated after the launch.
  - The three decode-temperature candidates `decode_temp005`, `decode_temp010`
    and `decode_temp020` have **no card**: they were evaluated at [1271], [1276]
    and [1281] (62.09%, 62.55%, 61.56% on all 1,319 problems) but the launches
    that created those directories are not in the digest. The same gap affects
    `rft_decode_temp010` inside exp-16, which the digest shows created by a plain
    `cp` with no generation-config edit.
  - The command that populated `/home/ben/task/.vendor` with a working
    `liger_kernel` copy — the workaround that made exp-21 run at all — is not in
    the stream either.
