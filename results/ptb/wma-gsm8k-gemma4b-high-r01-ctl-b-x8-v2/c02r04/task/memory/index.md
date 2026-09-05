# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.55 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.68 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 2.9 | sft | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.8 | rft | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.76 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.37 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.772 | /home/ben/task/final_model |
| exp-07 | 7.57 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.782 | /home/ben/task/ckpts/exp-07-soup |
