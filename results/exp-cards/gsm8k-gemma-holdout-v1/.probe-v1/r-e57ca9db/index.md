# r-e57ca9db — experiment cards

Base model: `Qwen/Qwen3-1.7B-Base`. Benchmark: gsm8k. Budget: 10 h, 1x H100.
20 cards, one per launch that can be pointed at a shell event in the digest.
Cards are numbered in the order the narration places the launches; `launch_i` is
the digest index where that command text actually appears.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 604 | null | sft | base_model (`final_model`) | `data_clean/gsm8k_train_sft.jsonl` | 2e-6 / 0.5 ep | failed | none | inconclusive | iterate |
| exp-02 | 667 | null | sft | base_model (`final_model`) | `data_clean/mixed_balanced_32k.jsonl` | 1e-6 / 100 steps | completed | 0.70 @ official --limit 50 | contradicted | reject |
| exp-03 | 177 | null | sft | base_model (`final_model`) | `data_clean/benchmark_adapt_gsm_nothink_exact.jsonl` | 5e-7 / 100 steps | completed (no-op: weights byte-identical) | none | inconclusive | iterate |
| exp-04 | 636 | null | sft | base_model (`final_model`) | `data_clean/benchmark_adapt_gsm_nothink_exact.jsonl` | 5e-7 / 100 steps | completed | 0.796875 @ dev-64 exact-fewshot (ckpt-50) | inconclusive | reject |
| exp-05 | 655 | null | decode-config | exp-04 | — | — | completed | 0.72 @ official --limit 50 | contradicted | reject |
| exp-06 | 1042 | null | rft | base_model (`final_model`) | `data_clean/rft_benchmark_exact_greedy_1000_filtered.jsonl` (515) | 5e-7 / 64 steps | completed | 0.8125 @ dev-64 exact-fewshot (ckpt-48) | inconclusive | reject |
| exp-07 | 1062 | null | decode-config | exp-06 | — | — | completed | 0.72 @ official --limit 50 | contradicted | reject |
| exp-08 | 1088 | null | other (tokenizer artifact) | base_model (`final_model`) | — | — | completed | 0.68 @ official --limit 50 | contradicted | reject |
| exp-09 | 1129 | null | rft | base_model (`final_model`) | `data_clean/rft_plain_greedy_1000_filtered.jsonl` | 5e-7 / 72 steps | completed | none | inconclusive | reject |
| exp-10 | 1139 | null | decode-config | exp-09 | — | — | completed | 0.68 @ official --limit 50 | contradicted | reject |
| exp-11 | 1140 | null | decode-config | exp-09 | — | — | completed | none (never evaluated) | inconclusive | abandon_line |
| exp-12 | 127 | null | merge | base_model (`final_model`) + `variants/rft_current_fixedgc` | — | alpha 0.25 / 0.50 | completed | 0.72 @ official --limit 50 (a025) | contradicted | reject |
| exp-13 | 555 | null | other (chat-template artifact) | base_model (`final_model`) | — | — | completed | 0.74 @ official --limit 50 | contradicted | reject |
| exp-14 | 503 | null | dpo | base_model (`final_model`) | `data_clean/dpo_current_exact_wrong_160.jsonl` (157) | 5e-7 / 40 steps | failed (checkpoint save) | none | inconclusive | iterate |
| exp-15 | 509 | null | dpo | base_model (`final_model`) | `data_clean/dpo_current_exact_wrong_160.jsonl` (157) | 5e-7 / 40 steps | completed | 0.76 @ official --limit 400 (vs 0.755) | supported | adopt |
| exp-16 | 117 | null | dpo | base_model (`final_model`) | `data_clean/dpo_current_exact_wrong_320.jsonl` (320) | 4e-7 / 50 steps | completed | 0.796875 @ dev-64 exact-fewshot | contradicted | reject |
| exp-17 | 135 | null | other (packaging) | exp-15 | — | — | completed | 0.76 @ official --limit 400 | supported | adopt (submission) |
| exp-18 | 93 | null | dpo | exp-17 | `data_clean/dpo_current_exact_wrong_cleaned_220.jsonl` (201) | 4e-7, beta 0.05 / 40 steps | completed | 0.70 @ official --limit 50 | contradicted | reject |
| exp-19 | 482 | null | dpo | exp-17 | `data_clean/dpo_self_current_exact_sampled_160_semiclean.jsonl` (96) | 4e-7, beta 0.05 / 30 steps | completed | 0.18 @ official --limit 50 (ckpt-15) | contradicted | reject |
| exp-20 | 1308 | null | other (tokenizer artifact) | exp-17 | — | — | completed | 0.752 @ official --limit 400 | contradicted | reject |

## Run-level notes

- **Adopted / submitted:** exp-17 packages exp-15's DPO checkpoint into
  `final_model`. Every later round through the last event ([2107]) re-confirms
  `final_model` unchanged, so exp-17 holds the submission.
- **Digest ordering defect.** The `say` events run in a clean chronological arc
  across turns 0-21, but the `shell` events are permuted against it: many sit at
  an index whose surrounding narration belongs to a different turn. Clear cases:
  [2] (turn=0) evaluates `variants/final_dpo_fixflag_v1`, a directory the
  narration only creates at [2101] (turn=20); [135] (turn=0) promotes the DPO run
  into `final_model`, which the narration does at [1701] (turn=11); [1308]
  (turn=7) creates that fixflag variant. Launches were therefore matched to
  problems/results by content, not by adjacency, and card order follows the
  narration.
- **`elapsed_h` and `wall_h` are null on every card.** The digest carries no
  event timestamps, only turn numbers, and `system_monitor.log` records PIDs
  without command lines, so no launch can be dated. The only clock statement in
  the stream is "about `1:26` left" at [2003] (turn=19).
- **Most launches are not in the digest.** The narration describes many training
  runs that never appear as a shell event and therefore get no card, including
  the main mixed-data run `runs/mixed_e03` that produced `checkpoint-400`, the
  turn-1 winner `refine_ckpt400_gsm_think_e05_lr2e6`, the alpha=0.010 soup that
  became `final_model` at turn 6, `refine_final_rft_current_exact_468_lr5e7`,
  `refine_final_exactgold_aligned_lr2e7`,
  `refine_final_rft_exact_disablethinking_502_lr5e7`,
  `sft_final_exact_wrong201_lr8e7`, `sft_final_exact_mix402_lr4e7`,
  `dpo_final_clean201_lr2e7_b002_s20`, the `refine_final_clean_gsm_think_..._rerun`,
  and every promotion into `final_model` except exp-17.
- **Consequence for chaining.** Cards exp-01 to exp-16 loaded `final_model`,
  which at those points was a derived checkpoint whose producing launch is absent
  from the digest, so `setup.parent_checkpoint.origin` falls back to
  `base_model` and `evaluation.comparator.ref` is written as `incumbent`. Only
  exp-17 to exp-20 chain to a real card.
- **No training or data-prep script is in the workspace snapshot**
  (`scripts/train_sft.py`, `scripts/train_dpo.py`, `scripts/build_*.py`,
  `scripts/eval_dev.py`, `scripts/prepare_math_data.py` are all absent), so no
  hyper-parameter was ever filled from an argparse default; unset fields are
  null. The DPO beta for exp-14/exp-15/exp-16 is unknown - no `--beta` was passed.
- `outcome.official_accuracy` is left null everywhere (written only on exp-17,
  as null).
