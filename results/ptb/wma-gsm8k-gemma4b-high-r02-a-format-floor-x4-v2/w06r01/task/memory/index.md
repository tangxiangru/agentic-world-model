# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.2 | other | base_model | closed | yes | supported | iterate | accuracy=0.0733 |  |
| exp-02 | 0.35 | sft | base_model | closed | yes (re-locked 2x) | supported | adopt | accuracy=0.6667 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 1.6 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.5933 | /home/ben/task/ckpts/exp-02/final |
| exp-04 | 1.9 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 5.4 | rft | exp-04 | closed | yes (re-locked 1x) | contradicted | reject | accuracy=0.7333 | /home/ben/task/ckpts/exp-05/final |
| exp-06 | 7.3 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.72 | /home/ben/task/ckpts/soup3 |
| exp-07 | 7.7 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.732 | /home/ben/task/ckpts/exp-04/final |
