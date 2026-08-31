# r-f238cc6e - reconstructed experiment cards

base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, 1x H100
digest has no per-event timestamps, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 143 | null | sft | base_model | openai/gsm8k train minus 256 holdout (7,217) | 1e-5 / 3 (stopped at step 800) | killed | 0.828 dev-64 (checkpoint-800) | inconclusive | reject |
| exp-02 | 283 | null | sft | base_model | math-ai/TemplateGSM templategsm-1000-1k (10,000) | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-03 | 301 | null | sft | base_model | math-ai/TemplateGSM templategsm-1000-1k (10,000) | 1e-5 / 1 | completed | none (final train loss 0.323, 625 steps) | inconclusive | adopt |
| exp-04 | 328 | null | sft | exp-03 | openai/gsm8k train minus 256 holdout | 1e-5 / 2 | killed | none | inconclusive | abandon_line |
| exp-05 | 333 | null | sft | exp-03 | openai/gsm8k train minus 256 holdout | 1e-5 / 2 | failed | none | inconclusive | abandon_line |
| exp-06 | 339 | null | sft | exp-03 | openai/gsm8k train minus 256 holdout | 1e-5 / 2 (stopped at step 800) | killed | 0.8125 dev-64 (both checkpoints) | contradicted | reject |
| exp-07 | 394 | null | sft | base_model | openai/gsm8k train, full split (7,473) | 1e-5 / 2 capped at 829 steps | completed | 0.000 official --limit 10 | inconclusive | adopt |
| exp-08 | 509 | null | decode-config | exp-07 | none | n/a | completed | 0.400 official --limit 10 | supported | adopt |
| exp-09 | 552 | null | decode-config | exp-08 | none | n/a | completed | 0.640 official --limit 50 (0.600 at --limit 10) | supported | adopt |
| exp-10 | 598 | null | sft | exp-09 | openai/gsm8k train (format-only continuation) | 5e-6 / 1 capped at 300 steps | failed | none | inconclusive | abandon_line |
| exp-11 | 609 | null | sft | exp-09 | openai/gsm8k train (format-only continuation) | 5e-6 / 1 capped at 150 steps | completed | 0.600 official --limit 10 | contradicted | reject |

Submission: `final_model` as it stands after **exp-09** - exp-07's weights plus the
dual-EOS (exp-08) and greedy-decoding (exp-09) metadata patches, verified at 0.640
on the official 50-sample check ([619], [620], [621]).

Notes
- Smoke runs are folded into exp-01's `provenance.smoke_runs`: the dev-evaluator
  probe at [135] and the 2-step trainer smoke at [136].
- exp-02/exp-04/exp-05 are immediate relaunch attempts of exp-03/exp-06 that never
  trained; they are cards because none was a deliberately truncated dry run.
- The metadata edits behind exp-08 and exp-09 are not shell events in the digest,
  so their `launch_i` points at the run that instantiated and measured the changed
  config; see each card's `provenance.unresolved`.
