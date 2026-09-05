# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.02 | /home/ben/task/ckpts/base_greedy |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 5.4 | rft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.8 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.8 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.8067 | /home/ben/task/final_model |
| exp-05 | 7.2 | other | exp-03 | closed | yes | contradicted | adopt | accuracy=0.7513 | /home/ben/task/final_model |
| exp-06 | 7.6 | merge | exp-03 | closed | yes | supported | adopt | accuracy=0.7867 | /home/ben/task/ckpts/soup_23 |
