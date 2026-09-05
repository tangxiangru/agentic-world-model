# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0467 |  |
| exp-02 | 0.65 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.3 | decode-config | exp-02 | closed | yes | contradicted | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 3.6 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.8 | /home/ben/task/ckpts/exp-02/ep1_greedy |
| exp-05 | 3.9 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.7733 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-06 | 6.6 | other | exp-05 | closed | yes | contradicted | adopt | accuracy=0.812 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-07 | 6.9 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7938 | /home/ben/task/ckpts/exp-05/final_greedy |
| exp-08 | 7.2 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7976 | /home/ben/task/final_model |
| exp-09 | 7.4 | merge | exp-05 | closed | yes | inconclusive | reject | accuracy=0.7998 | /home/ben/task/ckpts/soup_a05 |
| exp-10 | 8.0 | rft | exp-05 | closed | yes | contradicted | reject | accuracy=0.7733 | /home/ben/task/ckpts/exp-10/final_greedy |
