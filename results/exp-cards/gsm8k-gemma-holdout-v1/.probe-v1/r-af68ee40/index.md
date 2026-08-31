# r-af68ee40 — reconstructed experiment cards

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, one H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 72 | null | sft | base_model | gsm8k train | 1.5e-5 / 3 | completed | 0.560 @50 (checkpoint-500); 0.540 @150 | inconclusive | adopt |
| exp-02 | 133 | null | sft | exp-01 | gsm8k + synthetic 5k | 5e-6 / 1 | completed | 0.567 @150 | supported | adopt |
| exp-03 | 161 | null | sft | exp-02 | gsm8k + synthetic 8k | 3e-6 / 1 | killed | none | inconclusive | abandon_line |
| exp-04 | 184 | null | sft | exp-02 | gsm8k + synthetic 8k | 3e-6 / 1 | completed | 0.627 @150 (checkpoint-300) | supported | adopt |
| exp-05 | 270 | null | decode-config | exp-04 | none | null / null | completed | 0.693 @150 | supported | adopt |
| exp-06 | 297 | null | decode-config | multiple (exp-01, exp-02, exp-04, uncarded) | none | null / null | completed | 0.700 @150 (gsm_best_reanchor_gsm_lr2e6_half) | inconclusive | reject |
| exp-07 | 321 | null | sft | uncarded checkpoint | gsm8k train | 1e-6 / 0.5 | completed | 0.693 @150 | contradicted | reject |
| exp-08 | 334 | null | sft | exp-04 | gsm8k train | 2e-6 / 0.5 | completed | none stated | inconclusive | reject |
| exp-09 | 363 | null | other (packaging) | uncarded checkpoint | none | null / null | completed | 0.727 @150 | inconclusive | adopt |
| exp-10 | 371 | null | sft | exp-04 | gsm8k train | 2e-6 / 0.25 (seed 43) | killed | none | inconclusive | abandon_line |
| exp-11 | 417 | null | sft | exp-09 | gsm8k + orca 20k | 1e-6 / 0.5 | failed | none | inconclusive | abandon_line |
| exp-12 | 445 | null | sft | exp-09 | gsm8k + orca 20k | 1e-6 / 0.35 | completed | 0.713 @150 | contradicted | reject |
| exp-13 | 458 | null | sft | exp-12 | gsm8k train | 2e-6 / 0.25 | completed | none stated | inconclusive | reject |
| exp-14 | 472 | null | sft | exp-09 | gsm8k + metamath 20k | 1e-6 / 0.35 | completed | 0.740 @150 | supported | adopt |
| exp-15 | 480 | null | sft | exp-14 | gsm8k train | 2e-6 / 0.25 | completed | none stated | inconclusive | reject |
| exp-16 | 496 | null | sft | exp-09 | gsm8k + metamath 20k (GSM_AnsAug,GSM_Rephrased) | 1e-6 / 0.35 | completed | none stated | inconclusive | reject |
| exp-17 | 510 | null | sft | exp-09 | gsm8k + metamath 40k | 1e-6 / 0.25 | completed | none stated | inconclusive | reject |
| exp-18 | 522 | null | sft | exp-09 | gsm8k + metamath 20k | 1e-6 / 0.35 (seed 29) | completed | none stated | inconclusive | reject |
| exp-19 | 532 | null | sft | exp-09 | gsm8k + metamath 20k | 1e-6 / 0.25 | completed | 0.713 @150; 0.7066 @1319 | contradicted | reject |
| exp-20 | 200 | 8.2 | other (packaging) | exp-14 | none | null / null | completed | 0.740 @150; 0.7286 @1319 | inconclusive | adopt |
| exp-21 | 601 | null | decode-config | multiple (exp-14, exp-19, uncarded) | none | null / null | completed | 0.697 @1319 (rescored rival) | inconclusive | reject |
| exp-22 | 276 | null | sft | exp-20 | gsm8k train (--assistant-eot) | 5e-7 / 0.05 | completed | 0.720 @150 | contradicted | reject |
| exp-23 | 633 | null | merge | exp-14 (+ uncarded, alpha 0.90/0.10) | none | null / null | completed | 0.713 @150 | contradicted | reject |
| exp-24 | 690 | 9.2 | sft | uncarded checkpoint | gsm8k + metamath 20k | 1e-6 / 0.45 | completed | 0.7067 @150 | contradicted | reject |

## Notes

- **Submission**: `exp-20` — the 0.35-epoch MetaMath-GSM checkpoint of `exp-14` packaged into
  `final_model`, with the generation config stopping on both `<|end_of_text|>` and `<|im_end|>`.
  0.740 on the 150-sample screen, 0.7286 over all 1,319 test samples.
- **Card order**: the digest's event indices are not in launch order. A group of blocks in the
  turn=0 region (including [109], [116], [165], [195], [200], [211], [232], [255], [262], [276],
  [293], [302], [359]) names artefacts that are only created in the turn=1 region. Cards are
  numbered in the order the content implies; `provenance.launch_i` is always the index where the
  command actually appears.
- **elapsed_h**: the digest carries no event timestamps and no `timer.sh` output, so it is null
  except where the agent stated the remaining time in prose ([555], [697]).
- **Uncarded launches**: several checkpoints that shape the chain have no launch command anywhere in
  the digest and therefore get no card — `runs/gsm_best_reanchor_gsm_lr2e6_half`,
  `runs/gsm_ckpt484_reanchor_gsm_lr2e6_quarter`, `runs/gsm_ckpt484_reanchor_gsm_lr2e6_quarter_seed29`,
  `runs/soup_meta075_old025`, the first packaging of `final_model` from
  `runs/gsm_e2_synth5k_plus8k_lr3e6/checkpoint-300` (~[227]-[238]), and the in-place edit that added
  `<|im_end|>` to `final_model/generation_config.json` ([597]-[600]).
