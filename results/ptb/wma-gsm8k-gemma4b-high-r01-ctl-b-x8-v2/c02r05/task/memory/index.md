# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.7 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.5533 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.1 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6933 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 3.0 | rft | base_model | closed | yes | inconclusive | iterate | accuracy=0.6533 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.2 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.676 | /home/ben/task/ckpts/soup24-greedy |
| exp-06 | 5.3 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.686 | /home/ben/task/ckpts/exp-06-greedy |
| exp-07 | 6.6 | sft | exp-06 | closed | yes | contradicted | iterate | accuracy=0.696 | /home/ben/task/ckpts/exp-07-greedy |
| exp-08 | 8.1 | decode-config | exp-07 | closed | yes | supported | adopt | accuracy=0.7133 | /home/ben/task/ckpts/exp-07-greedy |
