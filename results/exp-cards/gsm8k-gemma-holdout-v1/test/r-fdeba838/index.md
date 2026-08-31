# r-fdeba838 — reconstructed experiment cards

Base model post-trained: `google/gemma-3-4b-pt` · benchmark gsm8k · 10 h budget · 1x H100.
16 cards, one per launch that can be pointed at in the digest. `official_accuracy` is left null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 197 | 0.27 | sft | base_model | gsm8k train (7473) | 1e-5 / 3 | completed | 0.467 @150 (eval_v1.json) | inconclusive | adopt |
| exp-02 | 414 | 0.91 | decode-config | exp-01 | — | — | completed | 0.460 @150 (eval_v1_greedy.json) | contradicted | adopt |
| exp-03 | 451 | 0.97 | sft | base_model | gsm8k train + MetaMathQA (mm_gsm 60k / mm_math 8k) | 2e-5 / 2 | completed | 0.553 @150 (eval_v2.json) | supported | adopt |
| exp-04 | 463 | 1.06 | other (packaging) | exp-02 | — | — | completed | none (final_model not evaluated in this state) | inconclusive | adopt |
| exp-05 | 563 | 3.87 | other (packaging) | exp-03 | — | — | completed | 0.480 @150 defaults (eval_final.json) | inconclusive | adopt |
| exp-06 | 592 | 3.91 | sft | base_model | gsm8k train + MetaMathQA (80k / 16k) | 2e-5 / 2 | failed | none (OOM at step 100, bs=16) | inconclusive | iterate |
| exp-07 | 614 | 4.51 | sft | base_model | gsm8k train + MetaMathQA (70k / 14k) | 2e-5 / 2 | failed | none (launch reaped, never ran) | inconclusive | iterate |
| exp-08 | 626 | 4.54 | sft | base_model | gsm8k train + MetaMathQA (65k / 13k), max_len 768 | 2e-5 / 2 | failed | none (launch reaped, never ran) | inconclusive | iterate |
| exp-09 | 636 | 4.55 | sft | base_model | gsm8k train + MetaMathQA (65k / 13k), 83k kept | 2e-5 / 2 | completed | 0.147 @150 (eval_v3.json) | contradicted | reject |
| exp-10 | 738 | 6.93 | decode-config | exp-09 | — | — | completed | 0.067 @150 (eval_v3_samp.json) | contradicted | reject |
| exp-11 | 766 | 7.12 | sft (continuation) | exp-09 | gsm8k train (7473) | 5e-6 / 2 | completed | 0.113 @150 (eval_v4.json) | contradicted | reject |
| exp-12 | 843 | 7.67 | rft (STaR) | exp-03 | data/star.jsonl (self-generated) + gsm8k train + MetaMathQA (15k / 0) | 2e-5 / 1 | completed | 0.467 @150 (eval_v5.json) | contradicted | reject |
| exp-13 | 912 | 8.55 | decode-config | exp-05 | — | — | completed | 0.433 @150 defaults (eval_final_t07.json) | contradicted | reject |
| exp-14 | 922 | 8.61 | decode-config | exp-05 | — | — | completed | 0.400 @150 defaults (eval_final_t10.json) | contradicted | reject |
| exp-15 | 936 | 8.69 | decode-config | exp-05 | — | — | completed | 0.460 @150 defaults (eval_final_greedystop.json) | contradicted | reject |
| exp-16 | 948 | 8.74 | decode-config | exp-05 | — | — | completed | 0.460 @150 defaults (eval_final_confirm.json); 0.480 at --max-connections 8 | supported | adopt |

Notes

- The submission is `final_model` = the exp-03 (v2) weights, packaged by exp-05, with the plain-greedy `generation_config.json` restored by **exp-16** — the card whose output is what the run ended with.
- Measurements marked "defaults" were run with `evaluate.py`'s own defaults (`--limit 150 --max-connections 2 --max-tokens 4000`); the others used `--limit 150 --max-connections 8`. The two are not comparable: the same weights scored 0.553 at mc 8 and 0.480 at mc 2.
- Smoke runs are recorded on the cards, not as cards: `[94]` (data-pipeline prefix/mask check) on exp-01 and `[445]` (MetaMath reformatting check) on exp-03.
- The workspace snapshot for this run contains only `evaluate.py`, `timer.sh`, `system_monitor.log` and the two judgement files — no training scripts and no `eval_*.json`, so every path in `setup`/`result` points at a file that exists only inside the run, and every number comes from the digest.
