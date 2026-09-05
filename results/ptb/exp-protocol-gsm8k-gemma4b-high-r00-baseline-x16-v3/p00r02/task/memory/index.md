# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | adopt | accuracy=0.0867 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6133 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.35 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.35 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7467 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.25 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.7933 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.5 | sft | exp-05 | closed | yes | inconclusive | iterate |  |  |
| exp-07 | 8.6 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.76 | /home/ben/task/ckpts/exp-07/final |
