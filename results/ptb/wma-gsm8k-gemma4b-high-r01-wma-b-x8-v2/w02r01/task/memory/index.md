# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | reject | accuracy=0.0533 |  |
| exp-02 | 0.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.64 | /home/ben/task/ckpts/exp-02/ep1 |
| exp-03 | 2.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03/ep2_greedy |
| exp-04 | 3.4 | rft | base_model | closed | yes | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-04/greedy |
| exp-05 | 5.6 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7667 | /home/ben/task/ckpts/exp-05/greedy |
| exp-06 | 6.3 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-04/greedy |
| exp-07 | 6.5 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.736 | /home/ben/task/ckpts/exp-07/soup |
| exp-08 | 6.7 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7354 | /home/ben/task/final_model |
