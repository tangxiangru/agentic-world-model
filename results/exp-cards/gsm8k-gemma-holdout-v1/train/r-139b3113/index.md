# r-139b3113 — extracted experiment cards

Base model: Qwen/Qwen3-1.7B-Base · benchmark: gsm8k · budget: 10 h, 1x H100.
The digest carries no timestamps in its block headers, so `elapsed_h` is null on
every card. Datasets are loaded in-script (no data files were written).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 88 | null | sft | base_model | MetaMathQA-GSM 240k + GSM8K train x3 22,419 (262,419) | 2e-5 / 2 | failed | — | inconclusive | abandon_line |
| exp-02 | 119 | null | sft | base_model | MetaMathQA-GSM 240k + GSM8K train x3 22,419 (262,419) | 2e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-03 | 138 | null | sft | base_model | MetaMathQA-GSM 240k + GSM8K train x3 22,419 (262,419) | 2e-5 / 2 | completed | — (4428 steps; loss 1.40 -> 0.17) | inconclusive | adopt |
| exp-04 | 293 | null | other (packaging) | exp-03 | — | — | killed | — | inconclusive | abandon_line |
| exp-05 | 311 | null | other (packaging) | exp-03 | — | — | completed | accuracy 0.060, n=50 | inconclusive | adopt |
| exp-06 | 334 | null | sft | base_model | same mixture truncated to 100,000 rows | 2e-5 / 1 | completed | — (3125 steps, packing off) | inconclusive | reject |
| exp-07 | 364 | null | other (packaging) | exp-06 | — | — | completed | accuracy 0.033, n=150 | inconclusive | reject |
| exp-08 | 382 | null | sft | base_model | MetaMathQA-GSM 240k + GSM8K train x3 22,419 (262,419) | 5e-5 / 1 | completed | — (8201 steps, packing off) | inconclusive | reject |
| exp-09 | 386 | null | other (packaging) | exp-03 | — | — | completed | — | inconclusive | adopt |
| exp-10 | 456 | null | other (packaging) | exp-08 | — | — | completed | accuracy 0.020, n=50 (eval_v3_50.json) | inconclusive | reject |
| exp-11 | 468 | null | other (packaging) | exp-03 | — | — | completed | — (same weights as exp-05: 0.060, n=50) | inconclusive | adopt |

Submitted candidate: exp-11 — `final_model` restored from `trained_model_v1`,
the exp-03 checkpoint. Every verdict is `inconclusive`: the base model was never
evaluated, so no card has a comparator measured under its own protocol except
exp-10, and that card carries no stated hypothesis.
