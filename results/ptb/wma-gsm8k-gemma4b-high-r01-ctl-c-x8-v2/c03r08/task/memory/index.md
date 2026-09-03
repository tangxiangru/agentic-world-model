# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.05 | other | base_model | closed | yes | contradicted | iterate | accuracy=0.0533 |  |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.62 | /home/ben/task/ckpts/exp-02/hf |
| exp-03 | 1.9 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-02/hf_greedy |
| exp-04 | 2.3 | sft | base_model | closed | yes | inconclusive | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-04/hf |
| exp-05 | 5.7 | rft | exp-04 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7733 | /home/ben/task/ckpts/exp-05/hf |
| exp-06 | 8.05 | merge | exp-05 | closed | yes | contradicted | reject | accuracy=0.7667 | /home/ben/task/ckpts/exp-05/hf |
