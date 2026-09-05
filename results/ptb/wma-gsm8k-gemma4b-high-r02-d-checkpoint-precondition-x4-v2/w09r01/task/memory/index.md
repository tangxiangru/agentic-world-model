# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6467 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-03/greedy |
| exp-04 | 1.9 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-04/greedy |
| exp-05 | 5.7 | rft | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.79 | /home/ben/task/ckpts/exp-05/greedy |
| exp-06 | 7.7 | merge | exp-05 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.774 | /home/ben/task/ckpts/exp-06/soup3 |
