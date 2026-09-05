# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.0467 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.2 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.8067 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 4.4 | rft | exp-03 | closed | yes (re-locked 2x) | inconclusive | adopt | accuracy=0.8267 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.6 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.8067 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.7 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.8267 | /home/ben/task/final_model |
| exp-07 | 8.5 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7933 | /home/ben/task/ckpts/exp-07/final |
