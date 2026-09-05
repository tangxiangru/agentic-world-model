# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.3 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6733 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.0 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.5 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-04_greedy |
| exp-05 | 4.7 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7467 | /home/ben/task/ckpts/exp-05_greedy |
| exp-06 | 6.1 | merge | exp-04 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7 | /home/ben/task/ckpts/soup_0405_greedy |
| exp-07 | 6.4 | other | exp-04 | closed | yes | supported | abandon_line | accuracy=0.7467 | /home/ben/task/ckpts/exp-04_greedy |
| exp-08 | 7.2 | rft | exp-04 | closed | yes | contradicted | iterate | accuracy=0.7733 | /home/ben/task/ckpts/exp-08_greedy |
| exp-09 | 8.0 | other | exp-04 | closed | yes | contradicted | adopt | accuracy=0.7475 | /home/ben/task/ckpts/exp-04_greedy |
