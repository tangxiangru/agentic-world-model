# r-c3036179 - reconstructed experiment cards

26 cards, one per launch found in the digest. `@150` is the evaluator's first 150 gsm8k test items,
`@1319` the full test set. "sampling" / "greedy" mark which generation config the model directory
carried at the time; the switch to explicit greedy decoding at exp-11 re-ordered the whole
checkpoint ranking, so scores before and after it are not comparable.

Submitted: **exp-25** packages **exp-21**'s weights (`run_fixed_lr15e6_mid_seed1234`, 0.705 on the
full test set) into `final_model`; exp-26 then adjusted the packaged generation cap.

Not cards: the pipeline smoke run at [59] (on exp-01) and the failed `rsync` copy at [486] (on
exp-10), both recorded as `provenance.smoke_runs`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 66 | 0.13 | sft | base_model | gsm8k-train x2 + 12k synthetic (short prompt) | 2.00e-5 / 2ep | completed | 0.160 @150 (after stop-token patch; 0.033 before) | inconclusive | reject |
| exp-02 | 125 | 0.54 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 1.00e-5 / 2ep | completed | 0.427 @150 | supported | adopt |
| exp-03 | 164 | 1.29 | sft | exp-02 | gsm8k-train 7,473 (fixed 10-shot) | 5.00e-6 / 2ep | completed | 0.453 @150 | supported | adopt |
| exp-04 | 230 | 2.08 | sft | exp-03 | gsm8k-train 7,473 + 6k synthetic (fixed 10-shot) | 2.00e-6 / 1ep | completed | 0.447 @150 | contradicted | reject |
| exp-05 | 274 | 2.68 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 2ep | completed | 0.685 @1319 greedy (0.487 @150 sampling, 0.647 @150 greedy) | supported | adopt |
| exp-06 | 317 | 3.34 | sft | exp-05 | gsm8k-train 7,473 (fixed 10-shot) | 5.00e-6 / 2ep | completed | 0.613 @150 greedy (0.493 @150 sampling) | supported | adopt |
| exp-07 | 362 | 4.01 | sft | exp-06 | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-6 / 2ep | completed | 0.540 @150 sampling | supported | adopt |
| exp-08 | 411 | 4.69 | sft | exp-07 | gsm8k-train 7,473 (fixed 10-shot) | 1.00e-6 / 2ep | completed | 0.627 @150 greedy (0.493 sampling) | contradicted | reject |
| exp-09 | 457 | 5.35 | sft | exp-07 | gsm8k-train 7,473 (fixed 10-shot) | 1.00e-6 / 1ep | completed | 0.500 @150 sampling | contradicted | reject |
| exp-10 | 490 | 5.72 | other | exp-07 | - | - | completed | 0.560 @150 sampling | inconclusive | adopt |
| exp-11 | 500 | 5.76 | decode-config | exp-10 | - | - | completed | 0.633 @150 greedy | supported | adopt |
| exp-12 | 507 | 5.79 | decode-config | exp-11 | - | - | completed | 0.487 @150 | contradicted | reject |
| exp-13 | 513 | 5.81 | decode-config | exp-12 | - | - | completed | 0.647 @150 greedy | supported | adopt |
| exp-14 | 554 | 6.03 | other | exp-05 | - | - | completed | 0.647 @150 (evaluator defaults) | supported | adopt |
| exp-15 | 599 | 6.09 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 1ep | killed | none (killed at ~step 11) | inconclusive | abandon_line |
| exp-16 | 612 | 6.13 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 1ep | completed | 0.703 @1319 (checkpoint-234) | supported | reject |
| exp-17 | 661 | 6.58 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 2ep | completed | 0.672 @1319 (checkpoint-234) | contradicted | reject |
| exp-18 | 737 | 7.45 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 1ep (stop @234) | completed | 0.685 @1319 | contradicted | reject |
| exp-19 | 754 | 7.71 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 1ep (stop @234) | completed | 0.687 @1319 | contradicted | reject |
| exp-20 | 800 | 7.99 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 2.00e-5 / 1ep (stop @234) | completed | 0.627 @150 | contradicted | reject |
| exp-21 | 832 | 8.19 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 1.50e-5 / 1ep (stop @234) | completed | 0.705 @1319 | supported | adopt |
| exp-22 | 858 | 8.45 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 1.25e-5 / 1ep (stop @234) | completed | 0.701 @1319 | contradicted | reject |
| exp-23 | 882 | 8.71 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 1.50e-5 / 1ep (stop @260) | completed | 0.647 @150 | contradicted | reject |
| exp-24 | 924 | 8.95 | sft | base_model | gsm8k-train 7,473 (fixed 10-shot) | 1.75e-5 / 1ep (stop @234) | completed | 0.680 @150 | inconclusive | reject |
| exp-25 | 947 | 9.15 | other | exp-21 | - | - | completed | 0.696 @1319 (packaged, evaluator defaults) | supported | adopt |
| exp-26 | 986 | 9.31 | decode-config | exp-25 | - | - | completed | 0.701 @1319 (packaged, cap 1200) | supported | reject |
