# r-5633966f — reconstructed experiment cards

Base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100.
The digest carries no event timestamps, so `elapsed_h` is null on every card.
Smoke run [185] (2 steps on a 64-line slice) is recorded on exp-01, not as a card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 199 | null | sft | base_model | data/gsm_plain.jsonl (7473) | 2e-4 / 1.0 | completed | 0.16 @ n=50 | inconclusive | adopt |
| exp-02 | 218 | null | merge | exp-01 | — | — / — | completed | none (eval could not serve the export) | inconclusive | abandon_line |
| exp-03 | 237 | null | merge | exp-01 | — | — / — | completed | 0.16 @ n=50 | inconclusive | adopt |
| exp-04 | 261 | null | sft | exp-03 | data/gsm_6shot.jsonl (7473) | 1e-4 / 1.0 | completed | 0.04 @ n=50 | contradicted | reject |
| exp-05 | 510 | null | merge | exp-04 | — | — / — | completed | 0.04 @ n=50 | contradicted | reject |
| exp-06 | 521 | null | sft | exp-03 | data/gsm_plain_metamath20k_v2.jsonl (27473) | 1e-4 / 1.0 | completed | 0.06 @ n=50 | contradicted | reject |
| exp-07 | 679 | null | merge | exp-06 | — | — / — | completed | 0.06 @ n=50 | contradicted | reject |
| exp-08 | 691 | null | sft | exp-03 | data/gsm_plain.jsonl (7473) | 5e-5 / 1.0 | completed | 0.00 @ n=50 | contradicted | reject |
| exp-09 | 743 | null | merge | exp-08 | — | — / — | completed | 0.00 @ n=50 | contradicted | reject |
| exp-10 | 753 | null | merge | exp-01 | — | — / — | completed | 0.327 @ n=150 | inconclusive | adopt |

## Notes

- The submitted model is **exp-10**: `final_model`, a re-merge of the exp-01
  adapter `runs/exp_plain_1ep`, validated at 0.327 (stderr 0.038) on the
  150-sample setting.
- Every 50-sample measurement is compared against exp-03 (0.16), the merged
  export of the same plain-SFT adapter. The base-model comparator was taken at
  `--limit 30` (0.0, `baseline_30.json`), so no card is `supported`/
  `contradicted` against the base model.
- Training cards and their merge cards report the same measurement, because the
  merge is the mechanical export of the adapter that was actually served
  (exp-01/exp-03, exp-04/exp-05, exp-06/exp-07, exp-08/exp-09).
- Only `baseline_30.json` and `final_model_eval150.json` survive in the
  workspace snapshot; the four 50-sample eval JSONs are cited by path but are
  not in `task/`, and their values come from the printed summary blocks (0.16
  from the agent's statement at [247] only).
