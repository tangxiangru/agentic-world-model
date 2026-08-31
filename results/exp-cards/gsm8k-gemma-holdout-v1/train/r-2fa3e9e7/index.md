# Reconstructed experiment cards

One card per launch, in launch order. `best measurement` is the agent's own strongest eval of that launch's output, with the sample count; `(greedy)` marks a measurement taken after a zero-temperature generation default was introduced at about hour 8.45, which is not comparable with the sampled numbers that drove the earlier selection.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 122 | 0.30 | sft | base_model | MetaMathQA GSM-types 160k | 1e-5 / 1.0 | completed | 0.745 @1319 (greedy) | inconclusive | adopt |
| exp-02 | 394 | 1.25 | sft | exp-01 | Orca-Math 20k | 5e-6 / 1.0 | completed | 0.480 @150 | contradicted | reject |
| exp-03 | 426 | 1.46 | sft | exp-01 | Orca-Math 20k | 1e-6 const / 1.0 | completed | 0.473 @150 | contradicted | reject |
| exp-04 | 444 | 1.68 | sft | exp-01 | gsm8k-train 2k | 5e-6 / 1.0 | killed | - | inconclusive | abandon_line |
| exp-05 | 452 | 1.72 | sft | exp-01 | gsm8k-train 2k | 5e-6 / 1.0 | completed | 0.387 @150 | contradicted | reject |
| exp-06 | 468 | 1.85 | sft | exp-01 | gsm8k-train 2k | 5e-7 const / 1.0 | completed | 0.467 @150 | contradicted | reject |
| exp-07 | 509 | 2.04 | sft | base_model | MetaMathQA mixed 100k @0 | 1e-5 / 1.0 | completed | 0.720 @150 (greedy) | inconclusive | adopt |
| exp-08 | 557 | 2.80 | merge | exp-01 + exp-07 | - | - | completed | 0.553 @150 | inconclusive | reject |
| exp-09 | 564 | 2.84 | merge | exp-01 + exp-07 | - | - | completed | 0.719 @1319 (greedy) | supported | reject |
| exp-10 | 564 | 2.84 | merge | exp-01 + exp-07 | - | - | completed | 0.553 @150 | inconclusive | reject |
| exp-11 | 574 | 2.92 | merge | exp-01 + exp-07 | - | - | completed | 0.607 @150 | supported | reject |
| exp-12 | 574 | 2.92 | merge | exp-01 + exp-07 | - | - | completed | 0.607 @150 / 0.160 @500 | contradicted | reject |
| exp-13 | 574 | 2.92 | merge | exp-01 + exp-07 | - | - | completed | 0.633 @150 / 0.602 @500 | supported | adopt |
| exp-14 | 574 | 2.92 | merge | exp-01 + exp-07 | - | - | completed | 0.736 @1319 (greedy) | contradicted | reject |
| exp-15 | 603 | 3.16 | sft | base_model | MetaMathQA mixed 100k @100k | 1e-5 / 1.0 | completed | 0.700 @150 (greedy) | supported | adopt |
| exp-16 | 644 | 3.90 | merge | exp-07 + exp-15 | - | - | completed | 0.573 @150 | contradicted | adopt |
| exp-17 | 644 | 3.90 | merge | exp-01 + exp-16 | - | - | completed | 0.740 @150 (greedy) | contradicted | reject |
| exp-18 | 644 | 3.90 | merge | exp-01 + exp-16 | - | - | completed | 0.747 @150 (greedy) | inconclusive | reject |
| exp-19 | 644 | 3.90 | merge | exp-01 + exp-16 | - | - | completed | 0.738 @1319 (greedy) | contradicted | reject |
| exp-20 | 691 | 4.10 | sft | base_model | MetaMathQA mixed 100k @200k | 1e-5 / 1.0 | completed | 0.733 @150 (greedy) | contradicted | adopt |
| exp-21 | 737 | 4.22 | merge | exp-01 + exp-15 | - | - | completed | 0.620 @150 / 0.594 @500 | contradicted | reject |
| exp-22 | 962 | 4.81 | sft | base_model | MetaMathQA GSM-derived 99k (unpacked) | 1e-5 / 1.0 | completed | 0.060 @150 | contradicted | adopt |
| exp-23 | 1140 | 7.30 | sft | exp-22 | MetaMathQA GSM-types 20k | 1e-6 const / 1.0 | completed | 0.373 @150 | inconclusive | adopt |
| exp-24 | 1164 | 7.47 | sft | exp-23 | MetaMathQA GSM-types 40k @20k | 2e-6 const / 1.0 | completed | 0.427 @150 (greedy) | contradicted | reject |
| exp-25 | 1197 | 7.79 | merge | exp-01 + exp-20 | - | - | completed | 0.723 @1319 (greedy) | contradicted | reject |
| exp-26 | 1208 | 7.87 | sft | exp-13 | MetaMathQA mixed 20k @300k | 5e-7 const / 1.0 | completed | 0.727 @150 (greedy) | contradicted | reject |
| exp-27 | 1229 | 8.07 | merge | exp-01 + exp-07 | - | - | completed | 0.727 @1319 (greedy) | contradicted | reject |
| exp-28 | 1230 | 8.08 | merge | exp-01 + exp-07 | - | - | completed | 0.747 @150 (greedy) | contradicted | reject |
| exp-29 | 1252 | 8.16 | merge (packaging) | exp-01 + exp-07 | - | - | completed | 0.726 @1319 (greedy) / 0.061 @1319 (sampled) | contradicted | reject |
| exp-30 | 1375 | 8.83 | merge | exp-01 + exp-07 | - | - | failed | - | inconclusive | abandon_line |
| exp-31 | 1383 | 8.85 | merge | exp-01 + exp-07 | - | - | completed | 0.736 @1319 (true greedy) | contradicted | reject |
| exp-32 | 1383 | 8.85 | merge | exp-01 + exp-07 | - | - | completed | 0.760 @150 | inconclusive | reject |
| exp-33 | 1383 | 8.85 | merge | exp-01 + exp-07 | - | - | completed | 0.747 @150 | contradicted | reject |
| exp-34 | 1383 | 8.85 | merge | exp-01 + exp-07 | - | - | completed | 0.727 @150 | contradicted | reject |
| exp-35 | 1383 | 8.85 | merge | exp-01 + exp-07 | - | - | completed | 0.740 @150 | contradicted | reject |
| exp-36 | 1397 | 9.05 | other (packaging) | exp-01 | - | - | completed | 0.745 @1319 / 0.740 @150 stock | supported | adopt |
| exp-37 | 1464 | 9.50 | sft | exp-01 | MetaMathQA GSM-types 20k @160k | 2e-6 / 1.0 | failed | - | inconclusive | abandon_line |
| exp-38 | 1472 | 9.51 | sft | exp-01 | gsm8k-train 7k | 5e-7 / 1.0 | killed | - | inconclusive | abandon_line |
