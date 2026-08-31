# r-01ed5927 — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 102 | null | sft | base_model | sft_gsm8k.jsonl (7473, openai/gsm8k train) | 1e-5 / 3 | completed | 0.4267 @ n=150 (eval_sft_v1.json) | inconclusive | adopt |
| exp-02 | 297 | null | decode-config | exp-01 | none | null / null | completed | 0.8133 @ n=150 (eval_sft_v1_full.json) | supported | adopt |
| exp-03 | 336 | null | other (packaging) | exp-02 | none | null / null | completed | none | inconclusive | adopt |
| exp-04 | 663 | null | other (packaging) | sft_v3 (no card) | none | null / null | completed | 0.84 @ n=150 (eval_final_default.json) | inconclusive | adopt |

Run-level notes:

- exp-04 is the submission: final_model = the checkpoint copied in at [663], validated at
  84.0% under the eval's default command and 83.85% on the full 1319-item test set.
- The digest carries no timestamps, so `elapsed_h` is null on every card.
- Four further training launches are visible only by their side effects (log files
  train_v2.log, train_v3.log, train_v3b.log, train_v4.log, checkpoints sft_v2/sft_v3/sft_v4
  and the evals of them). Their launch commands are absent from the digest — the event
  stream jumps [352]→[404], [528]→[534], [574]→[580] and [684]→[688] — so no card is
  written for them, and exp-04's parent has no originating card.
