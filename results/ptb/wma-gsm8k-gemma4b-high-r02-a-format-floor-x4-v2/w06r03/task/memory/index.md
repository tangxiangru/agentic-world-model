# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | reject | accuracy=0.07333 |  |
| exp-02 | 0.85 | sft | base_model | closed | yes | supported | adopt | accuracy=0.58 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.3 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.71333 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.25 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.72 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.7 | merge | exp-04 | closed | yes | contradicted | iterate | accuracy=0.738 | /home/ben/task/ckpts/exp-05_soup |
| exp-06 | 6.25 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.70811 | /home/ben/task/ckpts/exp-06_soup2 |
