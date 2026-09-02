# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.0333 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.798 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.0 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-04 | 5.1 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.826 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.35 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.82 | /home/ben/task/ckpts/exp-05/final |
