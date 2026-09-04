# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.028 | /home/ben/task/models/base_greedy |
| exp-02 | 1.1 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.712 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.5 | sft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7635 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.4 | sft | exp-03 | closed | yes | inconclusive | abandon_line | accuracy=0.7665 |  |
| exp-05 | 5.2 | rft | exp-03 | closed | yes | supported | adopt | accuracy=0.7748 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.1 | rft | exp-05 | closed | yes | contradicted | reject | accuracy=0.7574 | /home/ben/task/ckpts/exp-06/final |
