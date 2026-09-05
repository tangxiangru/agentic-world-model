# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.0467 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.5733 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.05 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 4.85 | sft | exp-02 | closed | yes | contradicted | iterate | accuracy=0.7767 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.55 | other | exp-04 | closed | yes | inconclusive | adopt | accuracy=0.75 | /home/ben/task/ckpts/exp-04-greedy |
| exp-06 | 7.15 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.762 | /home/ben/task/ckpts/exp-06-soup-greedy |
| exp-07 | 7.45 | other | exp-06 | closed | yes | contradicted | adopt | accuracy=0.7286 | /home/ben/task/final_model |
