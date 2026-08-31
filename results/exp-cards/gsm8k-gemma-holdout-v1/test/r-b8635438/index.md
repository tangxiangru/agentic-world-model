# r-b8635438 — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100
8 cards. Submitted checkpoint: exp-08 (`final_model`), accuracy 0.36 on 150 samples.
No per-event timestamps in the digest, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 49 | null | sft | base_model | gsm8k train (7473) | 1.5e-4 / 3 | failed | — | inconclusive | abandon_line |
| exp-02 | 57 | null | sft | base_model | gsm8k train (7473) | 1.5e-4 / 3 | killed | — | inconclusive | abandon_line |
| exp-03 | 92 | null | sft | base_model | gsm8k train (7473) | 2e-4 / 2 | completed | — (train_loss 0.538) | inconclusive | adopt |
| exp-04 | 108 | null | merge | exp-03 | — | — | completed | — (eval produced no score) | inconclusive | abandon_line |
| exp-05 | 126 | null | merge | exp-03 | — | — | completed | accuracy 0.1875 @ n=80 | inconclusive | reject |
| exp-06 | 140 | null | sft | base_model | gsm8k train (7473) | 1e-4 / 3 | completed | — (train_loss 0.531, 420 steps) | inconclusive | adopt |
| exp-07 | 163 | null | merge | exp-06 | — | — | completed | accuracy 0.300 @ n=80 (+0.1125 vs exp-05) | supported | adopt |
| exp-08 | 178 | null | other (package to final_model) | exp-07 | — | — | completed | accuracy 0.36 @ n=150 | inconclusive | adopt |

Not cards: the base-model baseline eval at [32] (accuracy 0.033, n=60, `baseline_metrics.json`),
which serves as the comparator on exp-01/exp-02/exp-04/exp-05. No smoke tests or dry runs
appear in this run — exp-01 and exp-02 were full-scale launches that crashed and were killed.
