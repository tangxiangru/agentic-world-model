# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.5 | decode-config | base_model | closed | yes | contradicted | reject | accuracy=0.02 | /home/ben/task/ckpts/base_greedy |
| exp-03 | 0.6 | decode-config | base_model | closed | yes | supported | adopt | accuracy=0.0267 | /home/ben/task/ckpts/base_greedy |
| exp-04 | 0.7 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.6 | rft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.7267 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.6 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.744 | /home/ben/task/ckpts/exp-05/final |
| exp-07 | 7.0 | rft | exp-05 | closed | yes | supported | adopt | accuracy=0.758 | /home/ben/task/ckpts/exp-07/final |
