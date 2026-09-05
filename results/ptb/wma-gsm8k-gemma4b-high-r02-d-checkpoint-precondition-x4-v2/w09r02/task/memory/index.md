# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.04666666666666667 |  |
| exp-02 | 0.33 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6866666666666666 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.25 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7533333333333333 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.5 | rft | exp-03 | closed | yes (re-locked 2x) | inconclusive | iterate | accuracy=0.7666666666666667 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.1 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.774 | /home/ben/task/ckpts/soup_e03e04 |
| exp-06 | 7.7 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.756 | /home/ben/task/ckpts/exp-06/final |
