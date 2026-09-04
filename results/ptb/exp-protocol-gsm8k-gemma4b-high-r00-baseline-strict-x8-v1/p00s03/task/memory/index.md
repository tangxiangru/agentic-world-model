# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.62 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.05 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.6333 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 1.2 | sft | base_model | closed | yes | supported | adopt | accuracy=0.728 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.5 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.728 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.6 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.662 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.6 | merge | exp-04 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/final_model |
| exp-08 | 7.8 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7096 | /home/ben/task/ckpts/exp-08/final |
