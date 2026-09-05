# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.66 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.2 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 3.5 | sft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.6 | rft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.74 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.7 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.766 | /home/ben/task/ckpts/exp-05/final |
