# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.07333333333333333 |  |
| exp-02 | 0.55 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6933333333333334 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.25 | sft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-04 | 3.55 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.7466666666666667 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.02 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.8066666666666666 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 8.25 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.8 | /home/ben/task/ckpts/exp-06/final |
