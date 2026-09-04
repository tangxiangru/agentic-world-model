# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes (re-locked 3x) | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.05 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.0 | rft | exp-03 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7533 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.0 | merge | exp-03 | closed | yes | inconclusive | reject | accuracy=0.74754 | /home/ben/task/ckpts/exp-05/soup |
