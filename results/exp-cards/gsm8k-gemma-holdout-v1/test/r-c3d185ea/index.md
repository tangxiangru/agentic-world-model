# r-c3d185ea — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 80GB
12 cards, launch order. `elapsed_h` is the `t=` on the launch event. Every measurement is
the agent's own eval; `official_accuracy` is left null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 127 | 0.27 | sft | base_model | sft_data (72419: gsm8k-train x3 + MetaMathQA GSM 50k) | 2e-5 / 1.0 | failed (OOM at batch 32) | — | inconclusive | abandon_line |
| exp-02 | 132 | 0.30 | sft | base_model | sft_data (72419) | 2e-5 / 1.0 | completed (2264 steps) | 0.23 @100 · eval_v1_100.json | inconclusive | adopt |
| exp-03 | 294 | 1.78 | sft | exp-02 | fewshot_data (14946: gsm8k-train x2, 2-6-shot prefixes) | 5e-6 / 1.0 | completed | 0.61 @100 · eval_v2_100.json | supported | adopt |
| exp-04 | 329 | 2.52 | other (package to final_model) | exp-03 | — | — | completed | — (weights identical to exp-03) | inconclusive | adopt |
| exp-05 | 338 | 2.53 | decode-config (greedy) | exp-04 | — | — | completed | 0.66 @150 · eval_greedy_150.json | inconclusive | adopt |
| exp-06 | 356 | 2.62 | sft | exp-03 | fewshot_data_v2 (34946: gsm8k x2 + 15k MetaMathQA GSM variants + 5k plain) | 5e-6 / 1.0 | failed (OOM at step 1) | — | inconclusive | abandon_line |
| exp-07 | 371 | 2.65 | sft | exp-03 | fewshot_data_v2 (34946) | 5e-6 / 1.0 | completed (final loss 0.15) | 0.666 @500 · eval_v3_500.log | inconclusive | adopt |
| exp-08 | 470 | 4.65 | other (package to final_model) | exp-07 | — | — | completed | 0.685 @200 · eval_final_sanity.log | inconclusive | adopt |
| exp-09 | 491 | 4.72 | sft | exp-07 | fewshot_data_v2 (34946) | 2e-6 / 0.5 | failed (checkpoint saved only config.json) | — | inconclusive | abandon_line |
| exp-10 | 539 | 5.53 | decode-config (drop temperature) | exp-08 | — | — | completed | 0.602 @500 · eval_final_final.log | inconclusive | reject |
| exp-11 | 541 | 5.53 | sft | exp-07 | fewshot_data_v2 (34946) | 2e-6 / 0.5 | completed | 0.578 @500 · eval_v4.log | contradicted | reject |
| exp-12 | 593 | 6.47 | decode-config (restore temperature) | exp-10 | — | — | completed | 0.670 @500 · eval_final_restored.log | inconclusive | adopt |

**Submission**: `exp-12` — final_model holds exp-07's weights (base → exp-02 SFT → exp-03 few-shot SFT →
exp-07 larger-mixture SFT) served with `do_sample: false` and `temperature: 0.0`, last measured at
0.670 on 500 samples.

**Not cards** (smoke tests, recorded as `provenance.smoke_runs`): 104, 109, 115, 120 (on exp-01);
277, 280, 288, 291 (on exp-03); 351, 353 (on exp-06).

**Run-level notes**: the run stops at t=+6.60 h of a 10 h budget, the last ~30 events being the agent
restating that the work is finished; the digest gives no reason. Several eval result blocks
([463], [476], [574], [586], [597]) are filtered out of the digest, so those numbers are read from the
`eval_*.log` files in the workspace. Three trainer logs (sft_train.log, sft_train3.log, sft_train4.log)
were overwritten by the relaunch that followed each crash, so no log survives for exp-01, exp-06 or exp-09.
