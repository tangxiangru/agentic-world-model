# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes | supported | reject | accuracy=0.065 |  |
| exp-02 | 0.5 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.77 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.1 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.815 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.25 | sft | exp-02 | closed | yes | supported | iterate | accuracy=0.82 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.4 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.8283 | /home/ben/task/ckpts/exp-05_soup |
| exp-06 | 6.5 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.8117 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 8.2 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.8226 | /home/ben/task/final_model |
