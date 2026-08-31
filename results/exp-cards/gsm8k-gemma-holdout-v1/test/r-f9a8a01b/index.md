# Reconstructed experiment cards - r-f9a8a01b (gsm8k, google/gemma-3-4b-pt, 10 h, 1x H100)

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 131 | 0.22 | sft | base_model | gsm8k train + MetaMathQA-GSM (67,473) | 2e-4 / 1.0 | completed | 0.220 @50 | inconclusive | adopt |
| exp-02 | 317 | 1.71 | merge | exp-01 | - | - | completed | 0.220 @50 | inconclusive | reject |
| exp-03 | 365 | 1.82 | sft | base_model | gsm8k train (7,473) | 1e-4 / 2.0 requested, 1 ran | completed | 0.320 @50 | supported | adopt |
| exp-04 | 528 | 2.76 | merge | exp-03 | - | - | completed | 0.320 @50 | supported | reject |
| exp-05 | 540 | 2.81 | sft | base_model | gsm8k train (7,473) | 1e-4 / 3.0 | completed | 0.513 @1319 | supported | adopt |
| exp-06 | 979 | 5.57 | merge | exp-05 | - | - | completed | 0.513 @1319 | supported | reject |
| exp-07 | 1005 | 5.67 | sft | exp-05 | gsm8k train (7,473) | 2e-5 / 2.0 | completed | 0.508 @1319 | inconclusive | adopt |
| exp-08 | 1299 | 7.52 | merge | exp-07 | - | - | completed | 0.508 @1319 | inconclusive | reject |
| exp-09 | 1337 | 7.62 | sft | exp-07 | gsm8k train (7,473) | 1e-5 / 1.0 | completed | 0.521 @1319 | inconclusive | adopt |
| exp-10 | 1462 | 8.55 | merge | exp-09 | - | - | completed | 0.521 @1319 | inconclusive | adopt |
| exp-11 | 1484 | 8.64 | merge | exp-07 (checkpoint-468) | - | - | completed | 0.523 @1319 | inconclusive | adopt |
| exp-12 | 1598 | 9.36 | other (package to final_model) | exp-11 | - | - | completed | 0.500 @50 | inconclusive | adopt |
| exp-13 | 1651 | 9.43 | merge | exp-05 (checkpoint-936) | - | - | completed | 0.457 @1319 | contradicted | reject |
| exp-14 | 1760 | 9.79 | merge (50/50 weight average) + package to final_model | exp-11 + exp-10, via an average with no launch event | - | - | completed | 0.537 @1319 | supported | adopt |

Submitted artifact: exp-14 (final_model = 50/50 weight average of the exp-11 and
exp-10 merged checkpoints, 0.5368 on --limit 1319). exp-12 held final_model
between t=+9.36h and t=+9.79h before being replaced.
