# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | other | base_model | closed | yes | supported | iterate | accuracy=0.03333 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.69333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.95 | decode-config | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.7 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.69333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.6 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.69333 | /home/ben/task/ckpts/soup_0202_04 |
| exp-06 | 5.75 | other | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.724 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-07 | 6.3 | other | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.7066 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-08 | 6.8 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.71039 | /home/ben/task/ckpts/exp-08/final |
