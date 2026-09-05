# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.07 | other | base_model | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.08 |  |
| exp-02 | 1.15 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.53 | /home/ben/task/work/sft_v1 |
| exp-03 | 3.15 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.67 | /home/ben/task/work/sft_v1_greedy |
| exp-04 | 4.2 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.625 | /home/ben/task/work/rft_v1 |
| exp-05 | 6.1 | rft | exp-02 | closed | yes (re-locked 1x) | contradicted | abandon_line | accuracy=0.645 | /home/ben/task/work/rft_v2 |
| exp-06 | 7.5 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.655 | /home/ben/task/final_model |
