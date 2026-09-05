# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.067 |  |
| exp-02 | 0.37 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6733 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.1 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.6725 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.0 | rft | exp-03 | closed | yes (re-locked 2x) | inconclusive | iterate | accuracy=0.677 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.0 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.6657 | /home/ben/task/ckpts/soup34 |
| exp-06 | 7.2 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7013 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.4 | sft | exp-04 | closed | yes (re-locked 1x) | inconclusive | reject | accuracy=0.7081 | /home/ben/task/ckpts/exp-07/final |
