# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.1 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5867 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 4.0 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6867 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 4.85 | rft | exp-02 | closed | yes | contradicted | reject | accuracy=0.6667 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 6.6 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.68 | /home/ben/task/ckpts/soup-0204-greedy |
| exp-06 | 6.85 | other | exp-03 | closed | yes | supported | adopt | accuracy=0.724 | /home/ben/task/ckpts/soup-0204-greedy |
| exp-07 | 8.0 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.721 | /home/ben/task/ckpts/soup-0204-w35-greedy |
