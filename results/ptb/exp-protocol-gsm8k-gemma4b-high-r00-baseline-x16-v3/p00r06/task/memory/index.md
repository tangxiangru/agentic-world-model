# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.9 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-04 | 4.6 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.67 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.3 | sft | exp-02 | closed | yes | supported | adopt | accuracy=0.704 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.3 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.704 | /home/ben/task/ckpts/soup245 |
| exp-07 | 6.5 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.672 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 7.7 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.706 | /home/ben/task/final_model |
| exp-09 | 7.8 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7089 | /home/ben/task/final_model |
| exp-10 | 8.0 | sft | exp-05 | closed | yes | supported | iterate | accuracy=0.714 | /home/ben/task/ckpts/exp-10/final |
| exp-11 | 8.9 | other | exp-10 | closed | yes | supported | adopt | accuracy=0.73 | /home/ben/task/final_model |
