# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.27 | sft | base_model | closed | yes | supported | adopt | accuracy=0.68 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.9 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.712 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 4.6 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.726 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 7.4 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.7127 | /home/ben/task/ckpts/exp-05-soup |
| exp-06 | 7.7 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.7134 | /home/ben/task/final_model |
