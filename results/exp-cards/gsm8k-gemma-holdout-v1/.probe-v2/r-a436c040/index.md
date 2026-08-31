# r-a436c040 — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
No timestamps in the stream, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 39 | null | sft | base_model | openai/gsm8k train, cap 3000 | 2e-4 / 1.0 | failed | — | inconclusive | iterate |
| exp-02 | 45 | null | sft | base_model | openai/gsm8k train, cap 3000 | 2e-4 / 1.0 | completed | — (adapter not evaluated) | inconclusive | adopt |
| exp-03 | 59 | null | merge | exp-02 | — | — | completed | accuracy 0.05 @ n=40 (vs base 0.175) | contradicted | reject |
| exp-04 | 91 | null | sft | base_model | openai/gsm8k train, cap 5000 | 1.5e-4 / 1.0 | completed | — (adapter not evaluated) | inconclusive | adopt |
| exp-05 | 97 | null | merge | exp-04 | — | — | completed | accuracy 0.35 @ n=40 (vs exp-03 0.05) | supported | reject |
| exp-06 | 109 | null | sft | base_model | openai/gsm8k train, full split | 1e-4 / 2.0 | completed | — (adapter not evaluated) | inconclusive | adopt |
| exp-07 | 124 | null | merge | exp-06 | — | — | completed | accuracy 0.425 @ n=40 (vs exp-05 0.35) | supported | adopt |
| exp-08 | 133 | null | sft | base_model | openai/gsm8k train, full split | 1e-4 / 2.0 (LoRA r=128) | killed | — | inconclusive | abandon_line |
| exp-09 | 156 | null | other (package to final_model) | exp-07 | — | — | completed | accuracy 0.4866666666666667 @ n=150 (no comparator) | inconclusive | adopt |
