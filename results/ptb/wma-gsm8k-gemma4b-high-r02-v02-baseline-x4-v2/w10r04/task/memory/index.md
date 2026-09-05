# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.0267 | /home/ben/task/models/gemma3_base_greedy |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7267 | /home/ben/task/ckpt/exp-02/final |
| exp-03 | 3.1 | rft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate | accuracy=0.699 | /home/ben/task/ckpt/exp-03/final |
| exp-04 | 4.05 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7134 | /home/ben/task/ckpt/exp-04/final |
| exp-05 | 6.5 | merge | exp-04 | closed | yes | contradicted | adopt | accuracy=0.7267 | /home/ben/task/ckpt/exp-05_soup |
| exp-06 | 6.9 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.74 | /home/ben/task/ckpt/exp-06/final |
