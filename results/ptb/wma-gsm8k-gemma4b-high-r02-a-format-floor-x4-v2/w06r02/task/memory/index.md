# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.06 |  |
| exp-02 | 1.0 | sft | base_model | closed | yes | inconclusive | iterate |  |  |
| exp-03 | 1.25 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.06 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 2.9 | sft | exp-03 | closed | yes | supported | adopt | accuracy=0.6733 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.2 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.6933 | /home/ben/task/ckpts/exp-04-greedy |
| exp-06 | 3.7 | sft | exp-04 | closed | yes | inconclusive | abandon_line |  |  |
| exp-07 | 4.8 | rft | exp-04 | closed | yes | supported | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 6.1 | other | exp-07 | closed | yes | contradicted | adopt | accuracy=0.736 | /home/ben/task/ckpts/exp-04-greedy |
| exp-09 | 6.4 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.718 | /home/ben/task/ckpts/exp-09-soup |
| exp-10 | 7.4 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.726 | /home/ben/task/ckpts/exp-10/final |
