# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7133 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.25 | decode-config | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.45 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.6267 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.6 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.732 | /home/ben/task/ckpts/exp-03-greedy |
| exp-06 | 4.0 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.718 | /home/ben/task/ckpts/exp-06-soup |
| exp-07 | 5.0 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.728 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 5.55 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.73 | /home/ben/task/ckpts/exp-08/final |
