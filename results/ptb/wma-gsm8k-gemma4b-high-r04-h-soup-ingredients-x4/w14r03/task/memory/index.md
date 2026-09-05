# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.05333 |  |
| exp-02 | 0.7 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.64667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.4 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.75333 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.7 | sft | base_model | closed | yes (re-locked 1x) | inconclusive | adopt | accuracy=0.77333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.1 | rft | exp-04 | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.82 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 8.2 | other | exp-05 | closed | yes | contradicted | adopt | accuracy=0.81333 | /home/ben/task/final_model |
