# r-b351b70e — extracted experiment cards

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 44 | 0.19 | sft (LoRA r=16, a=32) | base_model | openai/gsm8k train (7473) | 2e-4 / 3 | completed | 0.475 acc, n=200 | inconclusive | reject |
| exp-02 | 52 | 0.59 | sft (full FT) | base_model | openai/gsm8k train (7473) | 2e-5 / 5 | completed | 0.435 acc, n=200 | contradicted | reject |
| exp-03 | 84 | 1.13 | sft (LoRA r=64, a=128) | base_model | openai/gsm8k train (7473) | 1e-4 / 5 | completed | 0.545 acc, n=200 | supported | reject |
| exp-04 | 92 | 1.72 | sft (LoRA r=128, a=256) | base_model | openai/gsm8k train (7473) | 1e-4 / 5 | completed | 0.630 acc, n=200 | supported | reject |
| exp-05 | 106 | 2.31 | sft (LoRA r=256, a=512) | base_model | openai/gsm8k train (7473) | 1e-4 / 7 | completed | 0.745 acc, n=200 (0.702, n=1319) | supported | adopt |

Notes:

- Every launch trains from `Qwen/Qwen3-4B-Base`; no card is chained off another
  card's checkpoint. `final_model` is one directory that each launch overwrites,
  so only exp-05's weights survive the run.
- The agent stated no problem and no hypothesis before any launch; all six
  `stated_by_agent` flags are `false` and the only narration in the stream is the
  closing summary at [120].
- Five further launches are recorded as `provenance.smoke_runs` rather than as
  cards: [18], [22], [32], [36] (four retries of the exp-01 configuration, each
  dying inside `SFTTrainer` construction before a training step) and [82] (the
  exp-03 configuration dying on a broken pandas import). Two eval attempts,
  [58] and [64], died the same way and are noted on exp-02.
