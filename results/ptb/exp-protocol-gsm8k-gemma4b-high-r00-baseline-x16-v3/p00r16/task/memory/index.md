# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | adopt | accuracy=0.04 | /home/ben/task/base_gemma3_pt |
| exp-02 | 0.6 | sft | base_model | closed | yes | supported | adopt | accuracy=0.4667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.1 | sft | base_model | closed | yes | inconclusive | iterate |  |  |
| exp-04 | 1.05 | sft | base_model | closed | yes (re-locked 1x) | contradicted | iterate | accuracy=0.4 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 3.9 | decode-config | exp-04 | closed | yes | supported | adopt | accuracy=0.4667 | /home/ben/task/ckpts/exp-05/greedy |
| exp-06 | 3.85 | rft | exp-04 | closed | yes | supported | adopt | accuracy=0.4733 | /home/ben/task/ckpts/exp-06/greedy |
| exp-07 | 5.2 | sft | exp-06 | closed | yes | supported | adopt | accuracy=0.72 | /home/ben/task/ckpts/exp-07/greedy |
| exp-08 | 7.0 | sft | exp-07 | closed | yes | supported | adopt | accuracy=0.7333 | /home/ben/task/ckpts/exp-08/greedy |
