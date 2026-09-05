# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.4 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7133 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.35 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-05_soup |
| exp-06 | 5.4 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.736 | /home/ben/task/ckpts/exp-04/final |
| exp-07 | 5.6 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.721 | /home/ben/task/final_model |
| exp-08 | 5.7 | decode-config | exp-04 | closed | yes | contradicted | reject | accuracy=0.7195 | /home/ben/task/ckpts/exp-08_reppen |
| exp-09 | 6.8 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7104 | /home/ben/task/final_model |
