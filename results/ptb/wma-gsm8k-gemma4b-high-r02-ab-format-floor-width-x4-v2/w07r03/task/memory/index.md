# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | reject | accuracy=0.04 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.1 | sft | exp-02 | closed | yes (re-locked 2x) | inconclusive | adopt | accuracy=0.7367 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.6 | merge | exp-03 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7233 | /home/ben/task/ckpts/exp-04-soup |
| exp-05 | 6.1 | sft | exp-03 | closed | yes (re-locked 1x) | inconclusive | reject | accuracy=0.7333 | /home/ben/task/ckpts/exp-05/final |
