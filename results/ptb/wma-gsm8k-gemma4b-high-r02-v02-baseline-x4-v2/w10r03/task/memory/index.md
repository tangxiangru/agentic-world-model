# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.35 | other | base_model | closed | yes | supported | iterate | accuracy=0.08 |  |
| exp-02 | 0.45 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.58 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.17 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6933 | /home/ben/task/ckpts/exp-02/final |
| exp-04 | 3.4 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.3 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.8 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.6933 | /home/ben/task/ckpts/soup_0204 |
| exp-07 | 7.1 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.735 | /home/ben/task/ckpts/exp-05/final |
| exp-08 | 7.4 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7133 | /home/ben/task/final_model |
| exp-09 | 7.6 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.725 | /home/ben/task/ckpts/soup_0405 |
