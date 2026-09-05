# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | adopt | accuracy=0.0533 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.5 | sft | base_model | closed | yes | inconclusive | iterate |  |  |
| exp-03 | 1.1 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.06 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.3 | sft | exp-03 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.1 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-05-greedy |
| exp-06 | 5.2 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.6933 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 6.9 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.748 | /home/ben/task/final_model |
| exp-08 | 7.1 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-08/final |
| exp-09 | 7.9 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-09-soup |
