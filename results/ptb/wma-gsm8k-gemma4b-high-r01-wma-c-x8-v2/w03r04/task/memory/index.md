# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.1 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.6733 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-05 | 5.1 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7267 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 6.5 | other | exp-05 | closed | yes | contradicted | reject | accuracy=0.704 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-07 | 7.0 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/soup3 |
| exp-08 | 7.6 | merge | exp-02 | closed | yes | inconclusive | reject | accuracy=0.708 | /home/ben/task/ckpts/soup2 |
