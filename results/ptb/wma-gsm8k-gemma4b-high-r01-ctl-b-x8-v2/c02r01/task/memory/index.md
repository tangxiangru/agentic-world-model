# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.465 | /home/ben/task/ckpts/exp-02-bf16 |
| exp-03 | 0.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.635 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 0.7 | sft | base_model | closed | yes | supported | adopt | accuracy=0.725 | /home/ben/task/ckpts/exp-04-e2 |
| exp-05 | 3.8 | rft | exp-04 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/ckpts/exp-05-e2 |
| exp-06 | 4.8 | sft | exp-05 | closed | yes | contradicted | reject | accuracy=0.715 | /home/ben/task/ckpts/exp-06-e1 |
| exp-07 | 6.5 | other | exp-05 | closed | yes | contradicted | iterate | accuracy=0.7317 | /home/ben/task/ckpts/exp-06-e1 |
| exp-08 | 6.6 | merge | exp-06 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7233 | /home/ben/task/ckpts/exp-08-soup |
| exp-09 | 7.1 | rft | exp-06 | closed | yes | contradicted | reject | accuracy=0.7167 | /home/ben/task/ckpts/exp-09-e1 |
| exp-10 | 7.6 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/final_model |
