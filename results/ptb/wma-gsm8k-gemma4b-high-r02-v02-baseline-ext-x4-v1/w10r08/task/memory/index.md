# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.0567 |  |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.9 | other | exp-02 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.73 | /home/ben/task/ckpts/exp-02/final |
| exp-04 | 3.5 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.0 | sft | base_model | closed | yes | supported | adopt | accuracy=0.77 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.9 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7468 | /home/ben/task/final_model |
