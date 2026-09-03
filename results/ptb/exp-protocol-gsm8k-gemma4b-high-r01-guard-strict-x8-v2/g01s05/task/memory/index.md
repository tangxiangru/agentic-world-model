# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6533 | /home/ben/task/ckpts/exp-01/final |
| exp-02 | 2.3 | decode-config | exp-01 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-01/final_greedy |
| exp-03 | 3.2 | rft | exp-01 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.6 | sft | exp-01 | closed | yes | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.4 | sft | exp-04 | closed | yes | inconclusive | iterate |  |  |
| exp-06 | 7.0 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/exp-06/final |
