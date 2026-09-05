# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0467 |  |
| exp-02 | 0.3 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6133 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.1 | rft | exp-03 | closed | yes | supported | adopt | accuracy=0.6533 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.4 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.6 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.6533 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.7 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.682 | /home/ben/task/final_model |
