# r-d4a6c52f - gsm8k, Qwen/Qwen3-1.7B-Base, 10 h, 1x H100

21 cards. The digest carries no event timestamps, so every `elapsed_h` is null.
Comparators used across the run: base model 0.142 @120 (`--max-tokens 1200`) and
0.083 @120 (`--max-tokens 4000`). Submitted checkpoint: exp-21 (`final_model` =
the epoch-1 checkpoint of the 5e-6 cot run, merged).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 66 | null | sft | base_model | gsm8k train (7473) | 2e-4 / 4 | failed | - | inconclusive | iterate |
| exp-02 | 72 | null | sft | base_model | gsm8k train (7473) | 2e-4 / 4 | completed | - | inconclusive | adopt |
| exp-03 | 109 | null | merge | exp-02 | - | - | completed | - | inconclusive | reject |
| exp-04 | 112 | null | merge | exp-02 | - | - | completed | - | inconclusive | abandon_line |
| exp-05 | 173 | null | sft | base_model | gsm8k train (7473) | 5e-5 / 3.0 | completed | - | inconclusive | adopt |
| exp-06 | 247 | null | merge | exp-05 | - | - | completed | - | inconclusive | reject |
| exp-07 | 248 | null | merge | exp-05 | - | - | completed | - | inconclusive | abandon_line |
| exp-08 | 313 | null | sft | base_model | gsm8k train (7473) | 1e-5 / 1.0 | completed | - | inconclusive | adopt |
| exp-09 | 328 | null | merge | exp-08 | - | - | completed | 0.150 @120 (t400) | supported | adopt |
| exp-10 | 338 | null | sft | base_model | gsm8k train (7473) | 2e-5 / 1.0 | completed | - | inconclusive | adopt |
| exp-11 | 352 | null | merge | exp-10 | - | - | completed | - | inconclusive | reject |
| exp-12 | 374 | null | other (packaging) | exp-09 | - | - | completed | 0.133 @150 (t4000) | inconclusive | reject |
| exp-13 | 451 | null | sft | base_model | gsm8k train (7473, answer_only) | 1e-5 / 1.0 | completed | - | inconclusive | adopt |
| exp-14 | 461 | null | merge | exp-13 | - | - | completed | 0.087 @80 (t4000) | inconclusive | reject |
| exp-15 | 469 | null | sft | base_model | gsm8k train (7473, answer_only + 10-shot system) | 1e-5 / 1.0 | killed | - | inconclusive | abandon_line |
| exp-16 | 528 | null | sft | base_model | gsm8k train (7473) | 5e-6 / 2.0 | completed | - | inconclusive | adopt |
| exp-17 | 536 | null | merge | exp-16 | - | - | completed | 0.158 @120 (t4000) | supported | reject |
| exp-18 | 551 | null | merge | exp-16 | - | - | completed | 0.193 @150 (t4000) | supported | adopt |
| exp-19 | 557 | null | sft | base_model | gsm8k train (7473) | 1e-5 / 1.0 | completed | - | inconclusive | adopt |
| exp-20 | 562 | null | merge | exp-19 | - | - | completed | 0.125 @120 (t4000) | contradicted | reject |
| exp-21 | 605 | null | other (packaging) | exp-18 | - | - | completed | 0.175 @40 (t4000) | supported | adopt |

Smoke tests folded into cards rather than counted: [168] on exp-05; [266], [280],
[283], [291], [300], [303], [304], [308] on exp-08.
