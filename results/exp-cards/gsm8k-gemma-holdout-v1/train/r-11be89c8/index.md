# r-11be89c8 - reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base | benchmark: gsm8k | budget: 10 h, 1x H100.
Base-model reference: accuracy 0.1625 on `evaluate.py --limit 80` (baseline_metrics.json, [30]/[45]).
The digest has no event timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 106 | null | sft | base_model | openai/gsm8k train (minus 512 val), 10-shot system block | 1.5e-4 / 2.0 | killed | none (val eval_loss 0.3751 @ ckpt-400) | inconclusive | adopt |
| exp-02 | 192 | null | merge | exp-01 | - | - | completed | acc 0.0417 (n=120) | inconclusive | reject |
| exp-03 | 221 | null | sft | base_model | openai/gsm8k train (minus 512 val), simple system msg | 2e-4 / 3.0 | killed | none (val eval_loss 0.3873 @ ckpt-200) | inconclusive | adopt |
| exp-04 | 258 | null | merge | exp-03 | - | - | completed | acc 0.0833 (n=120) | inconclusive | reject |
| exp-05 | 276 | null | decode-config | exp-04 | - | - | completed | acc 0.0750 (n=40) | inconclusive | reject |
| exp-06 | 297 | null | sft | base_model | openai/gsm8k train (minus 512 val), plain text | 1e-4 / 1.5 | completed | none (val eval_loss 0.2282 @ ckpt-300) | inconclusive | adopt |
| exp-07 | 382 | null | merge | exp-06 | - | - | completed | acc 0.0500 (n=120) | inconclusive | reject |
| exp-08 | 387 | null | merge | exp-06 | - | - | completed | acc 0.0250 (n=40) | inconclusive | reject |
| exp-09 | 427 | null | sft | base_model | openai/gsm8k train (minus 512 val), empty system msg | 2e-5 / 1.0 | completed | none (val eval_loss 0.4210 @ ckpt-400) | inconclusive | adopt |
| exp-10 | 484 | null | merge | exp-09 | - | - | completed | acc 0.1000 (n=40) | inconclusive | reject |
| exp-11 | 485 | null | merge | exp-09 | - | - | completed | acc 0.0750 (n=40) | inconclusive | reject |
| exp-12 | 507 | null | sft | base_model | openai/gsm8k train, 512 rows, empty system msg | 1e-6 / 0.002 (1 step) | completed | none | inconclusive | adopt |
| exp-13 | 512 | null | merge | exp-12 | - | - | completed | acc 0.1250 (n=80), -0.0375 vs base | contradicted | reject |
| exp-14 | 525 | null | sft | base_model | openai/gsm8k train, 512 rows, empty system msg | 0.0 / 0.002 | completed | none | inconclusive | adopt |
| exp-15 | 532 | null | merge | exp-14 | - | - | completed | acc 0.2375 (n=80), +0.075 vs base | supported | adopt |

Notes:
- exp-15 is the submitted card: `final_model` ([560]). Its weights are bit-for-bit
  identical to the base model ([549]), so the +7.5 pts over the baseline is decode
  variance on a sampled 80-item eval, not a capability gain.
- `decision: adopt` on the training cards is mechanical: each one's checkpoint became
  the parent of a later card. Only exp-15's output became the submission.
- Verdicts are `inconclusive` wherever the candidate and the base-model comparator were
  measured under different `--limit` (120 or 40 vs the baseline's 80).
- Five pipeline smoke runs ([78], [84], [87], [88], [96]) are recorded on exp-01 under
  `provenance.smoke_runs`, not as cards.
