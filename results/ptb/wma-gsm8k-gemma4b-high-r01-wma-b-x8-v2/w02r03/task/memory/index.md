# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.08 |  |
| exp-02 | 1.4 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.5933 | /home/ben/task/ckpts/exp-02 |
| exp-03 | 2.65 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6533 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 4.5 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-04 |
| exp-05 | 6.6 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/exp-05 |
| exp-06 | 7.05 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/exp-06 |
| exp-07 | 7.4 | other | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-04 |
