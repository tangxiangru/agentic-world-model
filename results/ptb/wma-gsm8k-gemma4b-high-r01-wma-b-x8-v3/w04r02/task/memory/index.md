# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.3 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.9 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/vfinal_greedy |
| exp-04 | 2.0 | sft | base_model | closed | yes | inconclusive | iterate | accuracy=0.74 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 2.5 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/soup_all4 |
| exp-06 | 5.4 | rft | exp-04 | closed | yes | inconclusive | iterate | accuracy=0.7533 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 6.3 | other | exp-06 | closed | yes | contradicted | adopt | accuracy=0.768 | /home/ben/task/ckpts/exp-04/vfinal_greedy |
| exp-08 | 6.7 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7657 | /home/ben/task/final_model |
| exp-09 | 7.0 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7779 | /home/ben/task/ckpts/exp-09/final |
