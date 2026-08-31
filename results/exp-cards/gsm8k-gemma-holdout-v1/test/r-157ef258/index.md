# r-157ef258 — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 110 | 0.28 | sft (LoRA r=64) | base_model | meta-math/MetaMathQA, GSM-derived types | 2e-4 / 1 | killed | — | inconclusive | abandon_line |
| exp-02 | 122 | 0.29 | sft (LoRA r=64) | base_model | meta-math/MetaMathQA, GSM-derived types, 239,995 ex | 2e-4 / 1 | completed | — (eval could not load the checkpoint) | inconclusive | adopt |
| exp-03 | 152 | 5.45 | other (packaging: processor configs into final_model) | exp-02 | — | — | completed | accuracy 0.600 (n=10, stderr 0.163) | inconclusive | adopt |

Notes: exp-01 and exp-02 are the same command (`nohup python train.py > train.log 2>&1 &`); exp-01 was superseded ~1 min later by exp-02 after `num_proc=16` was removed from the dataset map/filter. The smoke test at [90] (5 steps, 100 examples) is recorded on both training cards as `provenance.smoke_runs`, not as a card. No base-model comparator exists: the baseline eval at [6] was killed by the 120 s bash timeout. `outcome.official_accuracy` is not written on any card.
