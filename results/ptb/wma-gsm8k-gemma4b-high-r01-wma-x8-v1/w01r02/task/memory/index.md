# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.05333 |  |
| exp-02 | 1.45 | sft | base_model | closed | yes | supported | adopt | accuracy=0.60667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.42 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.78667 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 2.5 | sft | base_model | closed | yes | inconclusive | iterate | accuracy=0.76667 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 5.25 | merge | exp-02 | closed | yes | contradicted | iterate | accuracy=0.74 | /home/ben/task/ckpts/exp-05-soup |
| exp-06 | 5.33 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.75739 | /home/ben/task/ckpts/exp-04-greedy |
| exp-07 | 7.35 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.74754 | /home/ben/task/ckpts/exp-07-greedy |
| exp-08 | 8.05 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.73768 | /home/ben/task/ckpts/exp-08-soup |
