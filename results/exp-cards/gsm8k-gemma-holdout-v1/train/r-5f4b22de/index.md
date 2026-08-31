# r-5f4b22de — reconstructed experiment cards

Base model Qwen/Qwen3-4B-Base, gsm8k, 10 h budget, one H100. Eight launches carry a
candidate: five training runs, one data-driven SFT rebuild, and two copies into the
submission path. Accuracies are the agent's own evals; `@N` is the `--limit`. The
digest's 233 recipe-bearing events end at t=+6.86h with ~3.1 h of budget unspent,
mid-wait on the full-set confirmation eval.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 132 | 0.21 | sft | base_model | sft_gsm8k.jsonl — gsm8k train gold (7473) | 1e-5 / 3 | completed | 0.7867 @150 (eval_v1_150.json) | inconclusive | iterate |
| exp-02 | 359 | 0.70 | rft | base_model | + rft.jsonl — self-sampled correct solutions (35572) | 1e-5 / 3 | completed | 0.840 @150 (eval_v2_150.json) | supported | adopt |
| exp-03 | 465 | 2.06 | grpo | exp-02 | gsm8k train prompts, reward = correct final number (300 steps) | 1e-6 / — | completed | 0.8667 @150 (eval_v3_150.json) | supported | adopt |
| exp-04 | 528 | 2.97 | grpo | exp-03 | gsm8k train prompts, same reward (350 further steps) | 1e-6 / — | completed | 0.8533 @150 (eval_v3b_150.json) | contradicted | reject |
| exp-05 | 582 | 3.82 | other (packaging) | exp-03 | — (copy only) | — / — | completed | — (never evaluated in this state) | inconclusive | adopt |
| exp-06 | 582 | 3.82 | sft | base_model | + metamath_gsm_40k.jsonl — MetaMathQA GSM subset (75572) | 1e-5 / 2 | completed | 0.8467 @150 (eval_v4sft_150.json) | inconclusive | adopt |
| exp-07 | 633 | 5.93 | grpo | exp-06 | gsm8k train prompts, same reward (300 steps) | 1e-6 / — | completed | 0.9267 @150 (eval_v4grpo_150.json) | supported | adopt |
| exp-08 | 688 | 6.82 | other (packaging) | exp-07 | — (copy only) | — / — | completed | — (full 1319 eval launched, never returned) | inconclusive | adopt |

Notes

- **exp-08 is the submission**: `final_model` last holds exp-07's checkpoint,
  copied at [688]. Its confirming full-set eval was launched in the same event and
  never returns inside the digest, so the submission has no measurement beyond
  exp-07's 0.9267 on 150 items.
- `final_model` is a moving target: exp-03's checkpoint from [582], exp-07's from
  [688]. Each card's `output_checkpoint` names the durable directory as well.
- The run's shape is two levers crossed. GRPO on the narrow base saturates
  (0.84 → 0.8667 → 0.8533, exp-03/exp-04); MetaMath diversity is neutral under
  greedy decoding (0.84 → 0.8467, exp-06); the two together jump to 0.9267
  (exp-07). Nothing in the run isolates why, and no pass@k was measured on the two
  SFT bases to test the stated mechanism.
- Every measurement is a single 150-item eval (stderr ≈ 2.1–2.9 pts). Three of the
  run's four decisions — keeping exp-03 over exp-04, calling MetaMath neutral,
  preferring exp-07 — turn on deltas at or inside that band; only exp-07's +6.0 pts
  clears it.
- The base-model probe (0.40 @50, `runs/baseline_50.json`, launched at [85]) is not
  a card: it produced no candidate. It is exp-01's comparator, at a different
  `--limit`.
- The rejection-sampling generation ([292]) and the two data-prep launches ([114],
  [308]/[319]) are not cards either: they produced training data, not candidates.
  They appear as `setup.data[].build_command` on the cards that consume them.
- Three GRPO smoke tests ([414], [437], [450]) are recorded on exp-03 as
  `provenance.smoke_runs`; the first two crashed (OOM on a 64-wide logits batch,
  then a generation-config validator error at save) and the third passed.
- The in-place greedy `generation_config.json` patch applied to exp-01's checkpoint
  at [217] is folded into exp-01 rather than carded: it changed no weights and
  produced no candidate distinct from `sft_v1`.
- `runs/` and `logs/` are absent from the workspace snapshot, so none of the eval
  json files these cards cite can be re-read; only the scripts survive.
