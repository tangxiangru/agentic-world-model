# r-a870812c — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 8 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 60 | 0.04 | sft | base_model | gsm8k main (train), 7473 | 2e-4 / 3 | killed | none (vanished at step 374/1404, no checkpoint) | inconclusive | abandon_line |
| exp-02 | 81 | 2.25 | sft | base_model | gsm8k main (train), 7473 | 2e-4 / 3 | completed | accuracy 0.180, n=150 (logs/…PByj4kxb….json) | inconclusive | adopt |
| exp-03 | 108 | 3.15 | sft | base_model | gsm8k main (train), 7473 | 1e-4 / 5 (3 done) | killed | none (checkpoint-1404, merge never ran) | inconclusive | adopt |
| exp-04 | 133 | 4.95 | merge | exp-03 | — | — / — | completed | accuracy 0.267, n=150 (logs/…CKaydr4y….json) | supported | adopt |
| exp-05 | 142 | 5.17 | sft | base_model | gsm8k main (train), 7473 | 5e-5 / 3 | killed | none (pkill at step 85/1404, too slow) | inconclusive | abandon_line |
| exp-06 | 153 | 5.35 | sft | base_model | gsm8k main (train), 7473 | 8e-5 / 3 (1 done) | killed | none (checkpoint-468, never merged or scored) | inconclusive | abandon_line |
| exp-07 | 193 | 6.46 | merge | exp-06 | — | — / — | failed | none (ValueError: no adapter_config.json at gemma_gsm8k_v4/checkpoint-1404) | inconclusive | abandon_line |
| exp-08 | 218 | 6.50 | merge | exp-03 | — | — / — | completed | none (limit-50 eval crashed: vLLM would not start; limit-150 eval never reported) | inconclusive | adopt |

Notes

- The submitted artifact is **exp-08**: `{dir}/final_model` at the end of the run carries the
  21:19 timestamps of the `restore_v2.py` launch at [218], a merge of
  `gemma_gsm8k_v2/checkpoint-1404` (exp-03's adapter). It was never scored itself; the run's
  only measured numbers are exp-02's 0.180 and exp-04's 0.267, the latter on an earlier merge
  of the same adapter.
- Four launches share the path `{dir}/final_model` (exp-02 writes it, exp-04 overwrites it,
  exp-07 empties it, exp-08 rebuilds it). Every `decision: adopt` here means "became the
  incumbent at that point", not "survived to the end".
- Smoke runs are not cards: the two crashed-at-construction launches of `train_gsm8k.py`
  at [39] (`SFTConfig` has no `max_seq_length`) and [48] (`SFTTrainer` has no `tokenizer`)
  are recorded on exp-01 as `provenance.smoke_runs`; neither took an optimizer step.
- One launch is missing from the filtered digest: `restore_v2.py` was run once before [218]
  (final_model shows 21:17 timestamps at [210] and is evaluated at [212]), but that event is
  not in the stream, so no card cites it. It is recorded in exp-08's `provenance.unresolved`.
- Only three of the six launches that could have produced a candidate ever reached a
  merged model, and only two candidates were ever scored. Four evals died with
  "Failed to start vLLM server" ([91], [188], [213], [224]) and two more ([161]/[164],
  [231]) never printed a result.
- No comparator for the base model exists: it was never evaluated, so exp-02's 0.180 -
  which the agent calls its "baseline" [242] - is itself a fine-tuned model.
- `{dir}/logs/` is not in the workspace snapshot, so neither eval json can be re-read; both
  accuracies come from the console summaries at [103] and [137].
- Every hyperparameter on every card comes from script text: not one launch in this run
  passes an argument.
