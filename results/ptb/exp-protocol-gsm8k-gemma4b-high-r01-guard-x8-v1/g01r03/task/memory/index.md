# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.75 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6833 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.0 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 2.1 | sft | base_model | closed | yes | supported | adopt | accuracy=0.83 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.7 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.8 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.6 | other | exp-04 | closed | yes | contradicted | reject | accuracy=0.8175 | /home/ben/task/ckpts/exp-04/final |
| exp-07 | 7.9 | merge | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.83875 | /home/ben/task/ckpts/exp-07_soup |
| exp-08 | 8.25 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.8467 | /home/ben/task/final_model |
