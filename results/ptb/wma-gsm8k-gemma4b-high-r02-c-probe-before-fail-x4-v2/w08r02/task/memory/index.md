# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.055 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/served |
| exp-03 | 3.4 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.73 | /home/ben/task/ckpts/exp-03/served |
| exp-04 | 4.2 | sft | exp-03 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.74 | /home/ben/task/ckpts/exp-04/served |
| exp-05 | 6.1 | other | exp-04 | closed | yes (re-locked 1x) | contradicted | adopt | accuracy=0.6998 | /home/ben/task/ckpts/soup34/served |
| exp-06 | 6.7 | merge | exp-05 | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.7058 | /home/ben/task/ckpts/soup234/served |
| exp-07 | 7.2 | other | exp-06 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.70129 | /home/ben/task/ckpts/soup234/served |
| exp-08 | 7.6 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.7133 | /home/ben/task/final_model |
