# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.0 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.66 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.8 | merge | exp-02 | closed | yes | supported | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-04/soup |
| exp-05 | 6.1 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.3 | merge | exp-04 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-06/soup |
| exp-07 | 7.5 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.708 | /home/ben/task/ckpts/exp-06/soup |
| exp-08 | 7.8 | other | exp-06 | closed | yes | contradicted | adopt | accuracy=0.7267 | /home/ben/task/final_model |
