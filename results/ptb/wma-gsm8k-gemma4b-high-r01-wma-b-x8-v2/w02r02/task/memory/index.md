# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.1 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.45 | other | exp-02 | closed | yes | inconclusive | reject | accuracy=0.725 | /home/ben/task/ckpts/exp-02/ep1_greedy |
| exp-05 | 6.55 | rft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.725 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 7.95 | other | exp-03 | closed | yes | contradicted | adopt | accuracy=0.738 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-07 | 8.75 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.738 | /home/ben/task/ckpts/soup |
