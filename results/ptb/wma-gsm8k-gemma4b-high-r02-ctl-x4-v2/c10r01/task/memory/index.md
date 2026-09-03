# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.7 | decode-config | exp-02 | closed | yes | contradicted | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.8 | rft | exp-02 | closed | yes | contradicted | adopt | accuracy=0.81 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.35 | sft | exp-04 | closed | yes | inconclusive | iterate |  |  |
| exp-06 | 7.45 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.82 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.65 | merge | exp-06 | closed | yes | supported | adopt | accuracy=0.832 | /home/ben/task/ckpts/soup_0406 |
