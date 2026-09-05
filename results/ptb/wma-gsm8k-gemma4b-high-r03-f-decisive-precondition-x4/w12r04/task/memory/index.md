# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.0867 |  |
| exp-02 | 0.55 | sft | base_model | closed | yes | supported | adopt | accuracy=0.66 | /home/ben/task/ckpts/exp-02_ep2_bf16 |
| exp-03 | 4.1 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7533 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 4.35 | sft | exp-02 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.8267 | /home/ben/task/ckpts/exp-04_greedy |
| exp-05 | 6.4 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.7867 | /home/ben/task/ckpts/exp-05_greedy |
| exp-06 | 8.2 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.8133 | /home/ben/task/final_model |
