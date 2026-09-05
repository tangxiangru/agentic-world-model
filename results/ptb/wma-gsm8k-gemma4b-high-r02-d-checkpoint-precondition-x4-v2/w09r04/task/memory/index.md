# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.47 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.25 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 4.2 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7333 | /home/ben/task/ckpts/exp-04/checkpoint-900 |
| exp-05 | 5.65 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7165 | /home/ben/task/final_model |
| exp-06 | 6.9 | merge | exp-04 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7013 | /home/ben/task/ckpts/exp-06-soup |
| exp-07 | 7.25 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7081 | /home/ben/task/ckpts/exp-07/checkpoint-460 |
