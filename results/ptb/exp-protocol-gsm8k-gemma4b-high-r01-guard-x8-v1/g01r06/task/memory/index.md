# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.075 |  |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.685 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.0 | rft | exp-02 | closed | yes (re-locked 3x) | inconclusive | iterate | accuracy=0.69 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.85 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.71 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.25 | merge | exp-03 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.71494 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.4 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.70508 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.5 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.70887 | /home/ben/task/final_model |
