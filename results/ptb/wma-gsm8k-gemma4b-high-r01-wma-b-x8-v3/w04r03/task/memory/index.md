# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.1 | other | base_model | closed | yes | supported | iterate | accuracy=0.0533 |  |
| exp-02 | 0.9 | sft | base_model | closed | yes | inconclusive | iterate |  |  |
| exp-03 | 1.5 | sft | base_model | closed | yes | inconclusive | iterate |  |  |
| exp-04 | 1.5 | sft | base_model | closed | yes | supported | adopt | accuracy=0.8267 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.0 | rft | exp-04 | closed | yes | contradicted | reject | accuracy=0.794 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 6.7 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.802 | /home/ben/task/ckpts/soup_abc |
| exp-07 | 6.7 | sft | exp-04 | closed | yes | contradicted | reject | accuracy=0.784 | /home/ben/task/ckpts/exp-07/final |
