# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | contradicted | iterate | accuracy=0.0867 |  |
| exp-02 | 0.9 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.15 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 2.07 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.8133 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.55 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.8067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.79 | rft | exp-04 | closed | yes | contradicted | adopt | accuracy=0.8067 | /home/ben/task/ckpts/exp-06/final |
