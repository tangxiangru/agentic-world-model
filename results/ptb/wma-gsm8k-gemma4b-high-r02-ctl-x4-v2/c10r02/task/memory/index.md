# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | adopt | accuracy=0.0467 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.8 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 4.0 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.15 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.65 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-06 |
| exp-07 | 7.9 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.7733 | /home/ben/task/final_model |
| exp-08 | 7.95 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.76 | /home/ben/task/ckpts/exp-08 |
