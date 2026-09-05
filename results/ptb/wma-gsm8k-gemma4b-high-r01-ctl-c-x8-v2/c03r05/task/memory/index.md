# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.05333 |  |
| exp-02 | 0.37 | sft | base_model | closed | yes | supported | adopt | accuracy=0.71333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.6 | rft | base_model | closed | yes | contradicted | iterate | accuracy=0.71333 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.9 | sft | exp-03 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-05 | 5.7 | sft | exp-03 | closed | yes | contradicted | reject | accuracy=0.69333 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.9 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.71333 | /home/ben/task/ckpts/soup23 |
| exp-07 | 7.1 | other | exp-03 | closed | yes | inconclusive | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-03/final |
| exp-08 | 7.6 | merge | exp-03 | closed | yes | contradicted | adopt | accuracy=0.748 | /home/ben/task/ckpts/soup35 |
