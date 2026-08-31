# r-e79c6a8d — extracted experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 17760 | 1.17 | sft | base_model | runs/sft_v1 | 1e-5 / 2 | failed | none (OOM before any step) | inconclusive | iterate |
| exp-02 | 17977 | 1.21 | sft | base_model | runs/sft_v1 | 1e-5 / 2 | completed | 0.9067 @150 (checkpoint-170); 0.8711 @1319 | supported | reject |
| exp-03 | 28270 | 4.24 | sft | base_model | runs/sft_v2 | 1e-5 / 1 | completed | 0.9000 @150; 0.8795 @1319 (runs/sft2/final) | inconclusive | adopt |
| exp-04 | 32291 | 6.34 | decode-config | exp-03 | — | — / — | completed | 0.8620 @1319 (temp 0.6 / top-p 0.95 / top-k 20) | inconclusive | reject |
| exp-05 | 32569 | 6.51 | decode-config | exp-04 | — | — / — | completed | none (both verification evals crashed the eval server) | inconclusive | adopt |
| exp-06 | 35285 | 6.88 | decode-config | exp-05 | — | — / — | completed | 0.8795 @1319, max 625 output tokens, 0 truncations | supported | adopt |
| exp-07 | 36424 | 7.09 | decode-config | exp-05 | — | — / — | completed | 0.8840 @1319 (shipped final_model); 0.8933 @150 default flags | inconclusive | adopt |

Baseline for reference: Qwen/Qwen3-4B-Base scored 0.400 at --limit 150 (runs/base_150.json, [11336]); it is the comparator on exp-01/exp-02 and is not a card (an eval, not a launch).
