# r-bfd319db — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
Base-model comparator: accuracy 0.075 on `--limit 40` (`baseline_40.json`, [18]).
No timestamps in this digest, so every `elapsed_h` is null.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 113 | null | sft (LoRA r=64) | base_model | openai/gsm8k train (7473) | 3e-5 / 3.0 | completed | 0.020 @ n=100 (via merged export) | inconclusive | reject |
| exp-02 | 131 | null | merge | exp-01 | — | — | completed | 0.020 @ n=100 | inconclusive | reject |
| exp-03 | 148 | null | sft (LoRA r=16) | base_model | openai/gsm8k train (7473) | 1e-5 / 1.0 | completed | 0.075 @ n=40 (via merged export) | contradicted | adopt |
| exp-04 | 154 | null | merge | exp-03 | — | — | completed | 0.075 @ n=40 (delta 0.000) | contradicted | adopt |
| exp-05 | 165 | null | sft (LoRA r=32, answer-only) | base_model | openai/gsm8k train (7473) | 2e-5 / 2.0 | completed | 0.050 @ n=40 (via merged export) | contradicted | reject |
| exp-06 | 178 | null | merge | exp-05 | — | — | completed | 0.050 @ n=40 (delta -0.025) | contradicted | reject |
| exp-07 | 187 | null | other (packaging to final_model) | exp-04 | — | — | completed | 0.075 @ n=40 (`final_model_eval40.json`) | supported | adopt |

Submitted artifact: `final_model` (exp-07), a copy of `runs/expB-merged` (exp-04),
i.e. the exp-03 LoRA adapter merged into the base weights. Its measured
40-sample accuracy ties the untuned base model at 0.075.

Not cards (smoke/dry runs, listed on exp-01 as `provenance.smoke_runs`): [40]
(crashed on `SFTConfig(max_seq_length=...)`), [54] (5 steps, 256 examples), [59]
(merge of that adapter), [93] (32 steps, 512 examples, reformatted targets), [98]
(merge of that adapter). The base-model baseline eval [18] is a comparator, not a
launch; the failed eval attempts at [100], [155] and [177] raced an unfinished
merge and are recorded on the affected cards.
