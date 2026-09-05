# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | iterate | accuracy=0.02666666666666667 |  |
| exp-02 | 0.8 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6866666666666666 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.6 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | adopt | accuracy=0.7066666666666667 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.1 | merge | exp-03 | closed | yes | contradicted | reject | accuracy=0.6866666666666666 | /home/ben/task/ckpts/soup23 |
| exp-05 | 5.2 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7466666666666667 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.9 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.74 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.2 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.7133333333333334 | /home/ben/task/ckpts/soup56 |
