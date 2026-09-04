# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | reject | accuracy=0.0533 |  |
| exp-02 | 0.65 | decode-config | base_model | closed | yes (re-locked 1x) | contradicted | adopt | accuracy=0.0233 | /home/ben/task/ckpts/base_gemma_greedy |
| exp-03 | 0.85 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5833 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 2.85 | sft | exp-03 | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.05 | rft | exp-04 | closed | yes | inconclusive | iterate | accuracy=0.71 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 8.4 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.7067 | /home/ben/task/ckpts/exp-06_soup |
