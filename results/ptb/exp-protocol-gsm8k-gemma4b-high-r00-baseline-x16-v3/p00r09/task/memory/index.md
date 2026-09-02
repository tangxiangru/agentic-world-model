# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.055 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes | supported | adopt | accuracy=0.605 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.45 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.59 | /home/ben/task/ckpts/exp-03/final |
| exp-04 | 3.75 | rft | base_model | closed | yes | contradicted | reject | accuracy=0.59 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.4 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.6202 | /home/ben/task/ckpts/exp-04/final |
| exp-06 | 5.85 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.6406 | /home/ben/task/ckpts/exp-06/final |
| exp-07 | 7.35 | sft | exp-06 | closed | yes | contradicted | reject | accuracy=0.6368 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 8.5 | other | exp-07 | closed | yes | supported | adopt | accuracy=0.6505 | /home/ben/task/ckpts/exp-06/final |
