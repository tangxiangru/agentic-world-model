# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.04 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-02 |
| exp-03 | 3.4 | rft | base_model | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.6867 | /home/ben/task/ckpts/exp-03 |
| exp-04 | 5.4 | merge | exp-03 | closed | yes | contradicted | adopt | accuracy=0.71 | /home/ben/task/ckpts/exp-04-soup2 |
| exp-05 | 6.3 | decode-config | exp-04 | closed | yes | contradicted | reject | accuracy=0.686 | /home/ben/task/ckpts/exp-05-reppen |
| exp-06 | 6.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.738 | /home/ben/task/ckpts/exp-06 |
| exp-07 | 8.9 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/final_model |
