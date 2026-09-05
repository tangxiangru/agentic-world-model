# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | decode-config | base_model | closed | yes | contradicted | iterate | accuracy=0.06 | /home/ben/task/ckpts/base_greedy |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.35 | other | exp-02 | closed | yes | supported | iterate | accuracy=0.66 | /home/ben/task/ckpts/exp-02/checkpoint-300 |
| exp-04 | 1.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.05 | sft | exp-04 | closed | yes | inconclusive | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 4.65 | merge | exp-05 | closed | yes | supported | adopt | accuracy=0.702 | /home/ben/task/ckpts/soup_45 |
| exp-07 | 4.8 | sft | exp-05 | closed | yes | supported | adopt | accuracy=0.722 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 6.4 | merge | exp-07 | closed | yes | contradicted | adopt | accuracy=0.71418 | /home/ben/task/ckpts/exp-07/final |
| exp-09 | 6.75 | sft | exp-07 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.70584 | /home/ben/task/ckpts/exp-09/final |
| exp-10 | 8.4 | other | exp-07 | closed | yes | contradicted | adopt | accuracy=0.71494 | /home/ben/task/ckpts/soup_79 |
