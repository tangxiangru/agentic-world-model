# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.0333 |  |
| exp-02 | 0.42 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.4 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.15 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.6933 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.2 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.64 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.4 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.6867 | /home/ben/task/ckpts/exp-06_soup |
| exp-07 | 6.8 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.67 | /home/ben/task/ckpts/exp-06_soup |
