# r-10e35756 — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 6 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 78 | null | sft | base_model | gsm8k main + socratic (train) | 2e-4 / 3 | completed | none (eval_loss 0.1251 @1200) | inconclusive | adopt |
| exp-02 | 304 | null | merge | exp-01 | — | — / — | completed | none (limit-120 eval crashed: vLLM would not start) | inconclusive | adopt |
| exp-03 | 314 | null | other (packaging: base processor files into final_model) | exp-02 | — | — / — | completed | accuracy 0.34, n=50 (exp1_50.json) | inconclusive | reject |
| exp-04 | 327 | null | merge (rebuild with base tokenizer + processor) | exp-01 | — | — / — | completed | accuracy 0.40, n=5 (final_smoke.json) | inconclusive | adopt |
| exp-05 | 348 | null | sft | base_model | gsm8k main (train) | 1.5e-4 / 4 | killed | none (eval_loss 0.3697 @200) | inconclusive | abandon_line |
| exp-06 | 399 | null | sft | base_model | gsm8k main (train), minimal prompt | 2e-4 / 4 | killed | none (killed before checkpoint-100) | inconclusive | abandon_line |

Notes

- `elapsed_h` is null on every card: the digest header states this run carries no event
  timestamps. Only `timer.sh` readings bound the timeline (9:59 remaining at [17] →
  8:28 at [441]).
- Smoke runs are not cards: the two `runs/sanity` launches at [65] and [72]
  (`--num-train-epochs 0.01`) are recorded on exp-01 as `provenance.smoke_runs`.
- No comparator exists anywhere in the run: the limit-50 baseline eval of the base model
  launched at [25] was killed at [34]/[36] as too slow and its output file was never
  written, so every verdict is `inconclusive`.
- `{dir}/final_model` is written by exp-02, patched by exp-03 and overwritten by exp-04;
  the three share one path. The file timestamps at [435] match the exp-04 rebuild, which
  is therefore the submitted artifact.
