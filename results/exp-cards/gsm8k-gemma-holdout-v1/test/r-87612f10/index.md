# r-87612f10 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 6 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 145 | null | sft (LoRA r=64) | base_model | gsm8k main (train minus 256-row dev) | 1e-4 / 3 | completed | accuracy 0.440, n=25 (exp_lora_e3_lr1e4_eval25.json; dev 0.5117, n=256) | supported | reject |
| exp-02 | 230 | null | sft (full, text stack) | base_model | gsm8k main (train minus 256-row dev) | 2e-5 / 3 | completed | accuracy 0.5273, n=256 (exp_full_e3_lr2e5_dev.txt; official 0.240, n=25) | inconclusive | adopt |
| exp-03 | 325 | null | sft (full, curriculum warm-up) | base_model | meta-math/MetaMathQA GSM subset, 50k of 240k | 1e-5 / 1 | completed | none (checkpoint never scored) | inconclusive | adopt |
| exp-04 | 454 | null | sft (full, specialization) | exp-03 | gsm8k main (train minus 256-row dev) | 1e-5 / 2 | completed | accuracy 0.4805, n=256 (exp_meta50k_gsm_e2_lr1e5_dev.txt; ckpt-226 0.4609) | contradicted | reject |
| exp-05 | 567 | null | sft (full, prompt-matched) | exp-02 | gsm8k main (train minus 256-row dev) + evaluator 10-shot system turn | 5e-6 / 1 | completed | accuracy 0.360, n=25 (exp_full_promptmatch_e1_lr5e6_eval25.json) | supported | adopt |
| exp-06 | 713 | null | sft (full, all-train continuation) | exp-05 | gsm8k main (all 7473 train rows) + evaluator 10-shot system turn | 2e-6 / 1 | completed | accuracy 0.4933, n=150 (final_eval150.json; 0.440, n=25) | supported | adopt |

Notes

- The submission is exp-06: `{dir}/final_model`, written directly by the training launch at [713]
  (there is no separate packaging or merge launch in this run) and confirmed as the export in the
  agent's closing report [872]. `outcome.official_accuracy` is null on every card.
- `elapsed_h` is null on every card: the digest header states this run carries no event timestamps.
  Only `timer.sh` readings bound the timeline — 9:59 remaining at [7], 9:47 at [137], 9:33 at [172],
  9:26 at [191], 8:12 at [402], 7:42 at [468], 7:19 at [575], 7:11 at [605], 6:55 at [663] — after
  which the agent stopped calling it, so the last ~3 h of the run are unanchored.
- Smoke runs are not cards: the two 1-step `smoke_lora` / `smoke_full` launches at [108] and [138]
  are recorded on exp-01 as `provenance.smoke_runs`. Every other command in the stream is a
  compile check, a dataset probe, an eval, or a file inspection.
- Two lines run through the six cards. The direct line base → exp-01 (LoRA) → exp-02 (full) →
  exp-05 (prompt-matched) → exp-06 (all-train) carries the submission; the curriculum line
  base → exp-03 (MetaMath 50k) → exp-04 (GSM8K specialization) was measured against exp-02 on the
  local dev split, lost by 4.7 pts, and was dropped at [580].
- Comparability caveat, carried on exp-02 and exp-05: the LoRA official slice at [226] used
  `--max-tokens 512`, while every later official eval ([701] onward) used the evaluate.py default of
  4000, which the agent only discovered at [643]. exp-01's 0.440 and exp-02's 0.240 are therefore
  two different protocols, and the agent's remark at [705] that the official evaluator "exposed a
  much bigger gap than the local dev run suggested" compares across that change.
- All promotion decisions in the second half rest on 25-sample official slices whose own stderr
  (0.087–0.101, recorded in the json files) is close to the deltas being acted on: 0.240 → 0.360 →
  0.440. Only final_model was ever run at n=150.
- The MetaMath data passed the agent's own GSM8K-test overlap check twice ([476] exact, [489]
  normalized): 0 test overlaps, 7390 train overlaps. `{dir}/meta_overlap_report.txt`,
  `{dir}/contamination_judgement.txt` and `{dir}/disallowed_model_judgement.txt` are in the
  workspace but no command in the stream writes them, and the first disagrees with [479] on the
  train-overlap count.
- exp-06 trains with `--train-on-all`, which folds the 256-row dev split used by exp-01 to exp-04
  into training; the local dev protocol is not held out for the submitted model, and the agent never
  re-ran it there.
