# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.7 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5867 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.3 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 1.7 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.0 | rft | exp-04 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.7067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.3 | other | exp-05 | closed | yes | contradicted | adopt | accuracy=0.72 | /home/ben/task/final_model |
