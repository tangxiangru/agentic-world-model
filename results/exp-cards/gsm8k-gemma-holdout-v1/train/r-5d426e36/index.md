# r-5d426e36 — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
Digest covers t=+0.00h .. t=+5.12h (events 0..357).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 109 | 0.61 | sft (LoRA r=32) | base_model | data/math_train_tokenized (OpenMathInstruct-2, 500000) | 2e-5 / 1 | killed | — | inconclusive | abandon_line |
| exp-02 | 126 | 0.81 | sft (LoRA r=64) | base_model | data/math_train_100k_tokenized (OpenMathInstruct-2, 100000) | 5e-5 / 3 (reached 1.5) | killed | — (final loss 0.315 @ step 3900) | inconclusive | adopt |
| exp-03 | 153 | 1.36 | merge | exp-02 (checkpoint-500) | — | — | completed | accuracy 0.380 (n=50, stderr 0.069) | inconclusive | adopt |
| exp-04 | 184 | 1.61 | merge | exp-02 (checkpoint-500) | — | — | completed | — (eval at --limit 150 timed out) | inconclusive | abandon_line |
| exp-05 | 233 | 3.54 | merge | exp-02 (checkpoint-3125) | — | — | failed | — | inconclusive | abandon_line |
| exp-06 | 256 | 3.80 | merge | exp-02 (checkpoint-3125) | — | — | failed | — | inconclusive | abandon_line |
| exp-07 | 302 | 4.44 | other (packaging copy) | exp-02 (checkpoint-3125) | — | — | completed (empty dir) | — | inconclusive | abandon_line |
| exp-08 | 304 | 4.46 | other (packaging copy) | exp-02 (checkpoint-3125) | — | — | failed | — | inconclusive | abandon_line |

Notes

- Only one accuracy number exists in the whole run: 0.380 on 50 gsm8k items for
  `final_model`, the merge of the step-500 LoRA adapter (exp-03,
  `eval_checkpoint500.log`). No baseline and no comparator was ever measured,
  so every verdict is `inconclusive`.
- exp-01 carries two pre-launch smoke runs of the same command
  (`provenance.smoke_runs`: [95] SFTConfig `max_seq_length`, [105] missing
  tensorboard).
- exp-05 .. exp-08 all target `checkpoints/math_ft_100k/checkpoint-3125`, which
  never existed: `--save-steps 500` only writes multiples of 500, and
  `save_total_limit=2` kept just the two latest.
- `outcome.official_accuracy` is deliberately absent from every card.
