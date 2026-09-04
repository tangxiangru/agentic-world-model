# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | other | base_model | closed | yes | supported | iterate | accuracy=0.0333 |  |
| exp-02 | 0.7 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.64 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.1 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.4 | other | exp-02 | closed | yes | inconclusive | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-02/ckpt1258_greedy |
| exp-05 | 4.95 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 6.3 | other | exp-05 | closed | yes | supported | reject | accuracy=0.7309 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-07 | 6.7 | merge | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7248 | /home/ben/task/ckpts/exp-07/soup |
| exp-08 | 7.0 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.7286 | /home/ben/task/final_model |
