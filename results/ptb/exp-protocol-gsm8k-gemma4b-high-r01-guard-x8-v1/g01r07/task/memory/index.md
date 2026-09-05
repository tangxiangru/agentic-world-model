# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.25 | other | base_model | closed | yes | supported | iterate | accuracy=0.08667 |  |
| exp-02 | 0.28 | sft | base_model | closed | yes | supported | adopt | accuracy=0.61333 | /home/ben/task/ckpts/exp-02/checkpoint-1820 |
| exp-03 | 2.65 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.60667 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 5.8 | rft | exp-03 | closed | yes | supported | adopt | accuracy=0.67333 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.4 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.656 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 8.05 | merge | exp-04 | closed | yes | supported | adopt | accuracy=0.664 | /home/ben/task/ckpts/exp-06-soup |
| exp-07 | 8.15 | decode-config | exp-06 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.722 | /home/ben/task/final_model |
| exp-08 | 8.35 | decode-config | exp-06 | closed | yes | supported | adopt | accuracy=0.736 | /home/ben/task/ckpts/exp-04/final |
