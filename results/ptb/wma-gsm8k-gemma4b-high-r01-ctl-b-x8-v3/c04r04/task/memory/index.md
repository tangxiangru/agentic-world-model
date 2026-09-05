# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0867 |  |
| exp-02 | 0.25 | sft | base_model | closed | yes | supported | adopt | accuracy=0.6667 | /home/ben/task/ckpts/exp-02/final_bf16 |
| exp-03 | 1.95 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.78 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.05 | rft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.7533 | /home/ben/task/ckpts/exp-04/final_greedy |
| exp-05 | 5.5 | other | exp-03 | closed | yes | contradicted | reject | accuracy=0.772 | /home/ben/task/ckpts/exp-03_greedy |
| exp-06 | 6.0 | merge | exp-02 | closed | yes | supported | adopt | accuracy=0.7867 | /home/ben/task/ckpts/exp-06_soup |
| exp-07 | 6.6 | other | exp-06 | closed | yes | contradicted | reject | accuracy=0.8 | /home/ben/task/final_model |
| exp-08 | 6.9 | other | exp-06 | closed | yes | supported | adopt | accuracy=0.7582 | /home/ben/task/final_model |
| exp-09 | 7.25 | sft | exp-06 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7399 | /home/ben/task/ckpts/exp-09/final_greedy |
