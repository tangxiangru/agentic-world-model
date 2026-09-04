# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.04667 |  |
| exp-02 | 0.25 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.62667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 1.65 | sft | base_model | closed | yes | contradicted | iterate | accuracy=0.63333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.3 | sft | exp-04 | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.77333 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.6 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.77333 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.8 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.774 | /home/ben/task/final_model |
