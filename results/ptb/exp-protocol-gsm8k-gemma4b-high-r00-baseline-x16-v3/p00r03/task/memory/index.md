# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes | supported | adopt | accuracy=0.367 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.2 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.6667 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.3 | rft | exp-03 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.9 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.708 | /home/ben/task/final_model |
| exp-06 | 7.4 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.674 | /home/ben/task/ckpts/exp-06/final |
