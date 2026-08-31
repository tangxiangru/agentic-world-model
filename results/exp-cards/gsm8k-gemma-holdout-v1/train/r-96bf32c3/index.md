# Reconstructed experiment cards — r-96bf32c3

Base model post-trained: `Qwen/Qwen3-4B-Base` · benchmark: gsm8k · budget: 10 h, 1x H100.
8 launches, in launch order. Adopted / submitted: **exp-07** (`final_model`, a copy of the exp-06 checkpoint).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 85 | 0.13 | sft | base_model | train_55k.jsonl (55,000 = gsm8k gold + MetaMathQA GSM_*) | 1e-5 / 1 | completed | 0.36 @ limit 50 (0.10 before the stop-token fix) | inconclusive | reject |
| exp-02 | 212 | 1.59 | sft | base_model | train_v2.jsonl (47,473 = 7,473 gold + 40k OMI train_1M) | 1e-5 / 1 | completed | 0.64 @ limit 50 · 0.58 @ limit 150 | supported | adopt |
| exp-03 | 261 | 2.56 | sft | base_model | train_v3.jsonl (112,419 = gold x3 + 90k filtered OMI train_1M) | 1e-5 / 1 | completed | 0.667 @ limit 150 | supported | adopt |
| exp-04 | 271 | 2.59 | other (packaging) | exp-02 | — | — | completed | — (final_model never evaluated) | inconclusive | adopt |
| exp-05 | 326 | 4.51 | other (packaging) | exp-03 | — | — | completed | — (final_model never evaluated) | inconclusive | adopt |
| exp-06 | 339 | 4.54 | sft (continuation) | exp-03 | train_v4.jsonl (67,365 = gold x5 + 30k fresh OMI train_1M, seed 44) | 5e-6 / 1 | completed | 0.70 @ limit 150 | supported | adopt |
| exp-07 | 381 | 5.76 | other (packaging) | exp-06 | — | — | completed | — (final_model never evaluated) | inconclusive | adopt |
| exp-08 | 391 | 5.78 | sft (continuation) | exp-06 | train_v5.jsonl (72,419 = gold x3 + 50k OMI train_2M, seed 100) | 3e-6 / 1 | completed | 0.66 @ limit 150 | contradicted | reject |

Four cards carry `adopt`: exp-04, exp-05 and exp-07 are the three successive writes to `final_model`,
and the training cards they package (exp-02, exp-03, exp-06) fed them. Only exp-07 survives to the end
of the run — it was written at [381] and the directory was verified unchanged at [493].

Comparators are recorded per card. exp-01 has none (the base model was never measured under any
protocol). exp-02's like-for-like comparison with exp-01 is at limit 50; exp-03, exp-06 and exp-08 are
each compared against their predecessor at limit 150. The exp-06 (+3.3 pts) and exp-08 (-4.0 pts)
deltas are both within about one standard error of a 150-item eval, and no candidate was ever
re-measured on more items.

Two smoke runs precede exp-01 (recorded on that card, not as cards of their own): [69] crashed on
`assistant_only_loss=True` with a chat template lacking `{% generation %}`, and [76] passed on 25 steps
after a training-only template was introduced.
