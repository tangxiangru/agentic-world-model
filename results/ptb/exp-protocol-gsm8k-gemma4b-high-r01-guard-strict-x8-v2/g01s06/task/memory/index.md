# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.4 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 6.7 | rft | exp-03 | closed | yes | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.8 | other | exp-03 | closed | yes | contradicted | iterate | accuracy=0.742 | /home/ben/task/ckpts/exp-03/final |
| exp-06 | 8.05 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.733 | /home/ben/task/ckpts/exp-03/final |
