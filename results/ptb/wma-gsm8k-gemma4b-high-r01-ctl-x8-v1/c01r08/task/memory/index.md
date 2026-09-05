# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.05333 |  |
| exp-02 | 0.73 | sft | base_model | closed | yes | supported | adopt | accuracy=0.69333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.95 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.68667 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.9 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 4.85 | rft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.76667 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.1 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.776 | /home/ben/task/ckpts/soup-235 |
| exp-07 | 6.6 | merge | exp-06 | closed | yes | supported | adopt | accuracy=0.786 | /home/ben/task/ckpts/soup-23 |
| exp-08 | 6.95 | merge | exp-07 | closed | yes | contradicted | adopt | accuracy=0.77082 | /home/ben/task/final_model |
