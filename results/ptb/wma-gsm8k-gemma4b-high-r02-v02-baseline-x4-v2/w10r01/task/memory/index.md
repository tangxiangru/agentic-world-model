# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.06 |  |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7067 | /home/ben/task/ckpts/exp-02/best |
| exp-03 | 3.4 | sft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-03/best |
| exp-04 | 5.5 | rft | exp-03 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.74 | /home/ben/task/ckpts/exp-04/best |
| exp-05 | 7.2 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.74 | /home/ben/task/final_model |
| exp-06 | 7.6 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.694 | /home/ben/task/ckpts/exp-04/best381 |
| exp-07 | 8.1 | merge | exp-04 | closed | yes | supported | adopt | accuracy=0.7 | /home/ben/task/ckpts/soup_final |
| exp-08 | 8.4 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.6983 | /home/ben/task/ckpts/soup_final |
