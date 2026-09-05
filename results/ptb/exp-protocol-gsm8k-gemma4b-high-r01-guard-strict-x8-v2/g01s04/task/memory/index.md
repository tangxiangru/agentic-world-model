# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6267 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.5 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-02/final_greedy |
| exp-04 | 5.4 | rft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-05 | 6.3 | other | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.75 | /home/ben/task/final_model |
