# Reconstructed experiment cards

- base model: google/gemma-3-4b-pt
- benchmark: gsm8k, 10 h budget, 1x H100
- cards: 9 (launches [121] .. [367]); smoke runs [99], [107], [110] are folded into exp-01
- no timestamps in this digest, so every `elapsed_h` is null

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 121 | null | sft | base_model | gsm8k/main train (7473) | 2e-4 / 2.0 | completed | none (adapter not scored) | inconclusive | adopt |
| exp-02 | 186 | null | merge | exp-01 | - | - / - | completed | accuracy 0.440 @ n=100 | inconclusive | adopt |
| exp-03 | 211 | null | sft | exp-02 | gsm8k/main (7473) + gsm8k/socratic + orca-math-word-problems-200k[:30000] | 1e-4 / 0.5 | completed | none (adapter not scored) | inconclusive | adopt |
| exp-04 | 319 | null | merge | exp-03 | - | - / - | failed | none | inconclusive | abandon_line |
| exp-05 | 325 | null | merge | exp-03 | - | - / - | completed | accuracy 0.410 @ n=100 (-0.030) | contradicted | reject |
| exp-06 | 333 | null | sft | exp-02 | gsm8k/main train, fixed 10-shot system block (1024 cap: 4096) | 5e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-07 | 338 | null | sft | exp-02 | gsm8k/main train, fixed 10-shot system block (1024) | 5e-5 / 1.0 | completed | none (adapter not scored) | inconclusive | adopt |
| exp-08 | 361 | null | merge | exp-07 | - | - / - | completed | accuracy 0.420 @ n=100 (-0.020) | contradicted | reject |
| exp-09 | 367 | null | merge | exp-01 | - | - / - | completed | accuracy 0.4067 @ n=150 | inconclusive | adopt |

Notes

- exp-09 is the submitted card: it wrote `final_model` and verified it from that
  path at `--limit 150`.
- `adopt` on exp-01, exp-03 and exp-07 means the output was carried forward as
  the parent of a later card, not that it became the incumbent; the branches
  from exp-03 and exp-07 were rejected on their merge cards (exp-05, exp-08).
- The only base-model score in the run is 0.000 at `--limit 20`, so no card has
  a base-model comparator under its own protocol.
