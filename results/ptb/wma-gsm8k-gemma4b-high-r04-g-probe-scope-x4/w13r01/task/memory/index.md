# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | decode-config | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.75 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.7081 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.6 | rft | exp-03 | closed | yes (re-locked 1x) | inconclusive | adopt | accuracy=0.7255 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.2 | rft | exp-04 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.702 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.2 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.724 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.1 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.7206 | /home/ben/task/ckpts/exp-04/final |
