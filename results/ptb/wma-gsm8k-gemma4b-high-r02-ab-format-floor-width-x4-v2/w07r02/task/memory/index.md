# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | other | base_model | closed | yes | supported | iterate | accuracy=0.0333 |  |
| exp-02 | 0.8 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.0 | decode-config | exp-02 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 4.5 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.4 | rft | exp-04 | closed | yes | supported | adopt | accuracy=0.7468 | /home/ben/task/ckpts/exp-05/final |
