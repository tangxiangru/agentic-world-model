# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.46 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 0.95 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.5867 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 1.55 | decode-config | exp-03 | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 1.65 | sft | base_model | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 4.25 | sft | exp-05 | closed | yes | inconclusive | iterate |  |  |
| exp-07 | 5.3 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.7667 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 6.75 | rft | exp-05 | closed | yes | supported | adopt | accuracy=0.7933 | /home/ben/task/ckpts/exp-08/final |
| exp-09 | 7.65 | other | exp-08 | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-08/final |
