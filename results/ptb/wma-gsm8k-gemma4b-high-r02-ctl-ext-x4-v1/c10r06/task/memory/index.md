# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.047 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.8 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.6 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.75 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 5.8 | merge | exp-02 | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.7333 | /home/ben/task/ckpts/exp-06-soup |
| exp-07 | 6.0 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.77 | /home/ben/task/ckpts/exp-06-soup |
| exp-08 | 6.4 | merge | exp-06 | closed | yes | contradicted | reject | accuracy=0.758 | /home/ben/task/ckpts/exp-08-soup4 |
| exp-09 | 6.6 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.7582 | /home/ben/task/final_model |
| exp-10 | 7.0 | other | exp-05 | closed | yes | supported | reject | accuracy=0.7468 | /home/ben/task/final_model |
