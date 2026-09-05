# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.4 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.5 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.4133 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.7 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.6 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.4 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.712 | /home/ben/task/ckpts/soup45 |
| exp-07 | 6.8 | sft | exp-05 | closed | yes | supported | adopt | accuracy=0.736 | /home/ben/task/ckpts/exp-07/final |
