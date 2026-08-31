# r-0af4240f — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, one H100.
The digest header states this run has no event timestamps, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 76 | null | sft | base_model | MetaMathQA GSM-only, 239,997 | 2e-5 / 2 | killed | none | inconclusive | abandon_line |
| exp-02 | 166 | null | sft | base_model | MetaMathQA GSM-only, 239,997 | 2e-5 / 2 | completed | 0.400 @ n=50 (eval_v2_quick.json) | inconclusive | adopt |
| exp-03 | 510 | null | decode-config | exp-02 | none | n/a | completed | 0.360 @ n=50 (eval_v2_fixed.json), −0.04 vs exp-02 | inconclusive | adopt |
| exp-04 | 531 | null | sft | exp-03 | MetaMathQA full, 392,670 | 1e-5 / 1 | completed | 0.473 @ n=150 (eval_v3_full.json); 0.480 @ n=50, +0.12 vs exp-03 | supported | adopt |
| exp-05 | 689 | null | other (packaging) | exp-04 | none | n/a | completed | 0.400 @ n=10 (eval_final_sanity.json) | inconclusive | adopt |

Submitted artifact: `/home/ben/task/final_model` = exp-05, the packaged exp-04 checkpoint.

## Run-level notes

- The filtered digest does not carry the shell events that launched the two
  completed training runs or the `eval_v2_fixed` evaluation. For exp-02 and
  exp-04, `provenance.launch_i` points at the write of the shell script that was
  run and whose content is the exact argv (`run_train_v2.sh` at [166],
  `run_continue.sh` at [531]); each card's `provenance.unresolved` records this
  and the evidence that the script did run.
- Two launches of the first training script that fought the trainer API are
  recorded as `provenance.smoke_runs` on exp-01 rather than as cards.
- The agent never evaluated the base model, so no card has a base-model
  comparator; exp-01's checkpoint was never produced and never scored.
