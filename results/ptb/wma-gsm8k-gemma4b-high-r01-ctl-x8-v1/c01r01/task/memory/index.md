# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.4 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.4 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.4 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7733 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.0 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.788 | /home/ben/task/ckpts/exp-03_greedy |
| exp-06 | 6.3 | decode-config | exp-02 | closed | yes | contradicted | reject | accuracy=0.778 | /home/ben/task/ckpts/exp-06_ck704_greedy |
| exp-07 | 6.3 | sft | exp-02 | closed | yes | supported | iterate | accuracy=0.806 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 7.4 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.7877 | /home/ben/task/ckpts/exp-07/final |
| exp-09 | 7.6 | merge | exp-07 | closed | yes | contradicted | reject | accuracy=0.7786 | /home/ben/task/ckpts/exp-09_soup |
