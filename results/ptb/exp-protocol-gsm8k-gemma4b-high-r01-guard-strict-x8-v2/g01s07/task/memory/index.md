# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | reject | accuracy=0.06 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7466666666666667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.8 | sft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-04 | 4.8 | sft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.7466666666666667 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.5 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.764 | /home/ben/task/ckpts/exp-02/final |
| exp-06 | 7.1 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.776 | /home/ben/task/ckpts/soup-0205 |
| exp-07 | 8.1 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.7407126611068992 | /home/ben/task/ckpts/soup-0205 |
