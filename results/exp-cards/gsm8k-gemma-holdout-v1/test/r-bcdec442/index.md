# r-bcdec442 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 12 cards

`{dir}` is the run's workspace, mirrored by the task snapshot at
`data/exp-cards/gsm8k-gemma-holdout-v1/task/r-bcdec442/`; `digests/r-bcdec442.md` is the
event stream at `data/exp-cards/gsm8k-gemma-holdout-v1/digests/r-bcdec442.md`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 79 | 0.19 | sft (LoRA r=128) | base_model | openai/gsm8k main (train), fixed 10-shot | 1.5e-4 / 3 | killed | none (killed before the first checkpoint) | inconclusive | abandon_line |
| exp-02 | 95 | 0.24 | sft (LoRA r=128) | base_model | openai/gsm8k main (train), fixed 10-shot | 1.5e-4 / 2 | completed | accuracy 0.613, n=150 (eval_lora_r128_e2_greedy_limit150.json) | inconclusive | adopt |
| exp-03 | 146 | 2.31 | merge | exp-02 | — | — / — | completed | accuracy 0.38, n=50 (eval_lora_r128_step500_limit50.json) | contradicted | reject |
| exp-04 | 151 | 2.36 | merge | exp-02 | — | — / — | completed | accuracy 0.40, n=50 (eval_lora_r128_step750_limit50.json) | contradicted | reject |
| exp-05 | 169 | 2.52 | sft (full text weights) | base_model | openai/gsm8k main (train), fixed 10-shot | 2e-5 / 2 | completed | accuracy 0.46, n=50 (eval_full_e2_limit50.json) | contradicted | reject |
| exp-06 | 195 | 4.16 | other (packaging: processor + greedy config into full-SFT checkpoints) | exp-05 | — | — / — | completed | accuracy 0.48, n=50 (eval_full_step500_limit50.json) | contradicted | reject |
| exp-07 | 201 | 4.21 | sft (LoRA r=128) | base_model | openai/gsm8k main (train), fixed 10-shot | 1.5e-4 / 3 | completed | accuracy 0.547, n=150 (eval_lora_r128_e3_limit150_maxtok4000.json) | contradicted | reject |
| exp-08 | 246 | 7.25 | merge | exp-07 | — | — / — | completed | accuracy 0.38, n=50 (eval_lora_r128_e3_step1000_limit50.json) | contradicted | reject |
| exp-09 | 255 | 7.34 | other (packaging: copy exp-02 to final_model) | exp-02 | — | — / — | completed | accuracy 0.46, n=150 (eval_final_model_limit150.json) | inconclusive | reject |
| exp-10 | 266 | 7.36 | sft (LoRA r=128, --add-plain-prompts) | base_model | openai/gsm8k main (train), 10-shot + plain prompts | 1.5e-4 / 1 | completed | accuracy 0.52, n=50 (eval_lora_r128_aug_plain_e1_b4_limit50.json) | contradicted | reject |
| exp-11 | 307 | 8.64 | other (packaging: copy exp-07 to final_model) | exp-07 | — | — / — | completed | accuracy 0.533, n=150 (eval_final_model_e3_greedy_limit150.json) | inconclusive | reject |
| exp-12 | 344 | 8.78 | decode-config (greedy config + copy exp-02 to final_model) | exp-02 | — | — / — | completed | accuracy 0.613, n=150 (eval_final_model_e2_greedy_limit150.json) | supported | adopt |

Notes

- The submitted artifact is **exp-12**: the exp-02 weights (2-epoch LoRA r=128, merged to
  full bf16) copied into `final_model` with an explicit greedy `generation_config.json`.
  exp-02 is also `adopt` because those are the weights; exp-12 is the packaging that
  became the submission.
- Smoke runs are not cards. Two LoRA pipeline checks ([66] crashed on an invalid
  GenerationConfig, [75] passed) sit on exp-01; three full-tuning batch-fit checks
  ([160], [163], [165]) sit on exp-05.
- Two protocols run through this stream and are not comparable. Every candidate was
  screened with `--max-tokens 1024` (`--limit 50`, then `--limit 150`); from [293] the
  agent switched to the evaluator's default `--max-tokens 4000`, which reordered the
  candidates and drove the final selection. `delta_vs_comparator` is null wherever a
  measurement and its comparator were taken under different settings.
- No base-model score exists at `--limit 50` or `--limit 150` — only 0.000 at
  `--limit 10` [34]. exp-01 and exp-02, whose comparator is the base model, are
  therefore `inconclusive` however large the lift.
- The decode-config change that produced the winning number has no card of its own: the
  edits that wrote an explicit greedy config into `final_model` (before [336]) and into
  `runs/lora_r128_e2_fixed` (before [340]) are not commands in the digest. Only their
  effect is visible, in the configs printed at [315] and [345]. It is folded into
  exp-12's setup and recorded in exp-02's and exp-11's `provenance.unresolved`.
- `{dir}/final_model` is written by exp-09, overwritten by exp-11 and overwritten again
  by exp-12; the three cards share one path.
- Never merged or evaluated, though retained: exp-02 checkpoint-250; exp-05
  checkpoint-250 (and checkpoint-750, packaged by exp-06 but never scored); exp-07
  checkpoints 250, 500, 750 and 1250; exp-10 checkpoint-500.
