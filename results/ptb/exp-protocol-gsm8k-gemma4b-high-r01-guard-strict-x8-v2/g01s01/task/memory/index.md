# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6133 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.7 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-03/greedy |
| exp-04 | 2.9 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.5 | rft | exp-03 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.3 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.7267 | /home/ben/task/ckpts/exp-06/soup3 |
| exp-07 | 7.5 | other | exp-03 | closed | yes | contradicted | adopt | accuracy=0.726 | /home/ben/task/ckpts/exp-03/greedy |
| exp-08 | 7.9 | other | exp-03 | closed | yes | supported | abandon_line | accuracy=0.76 |  |
