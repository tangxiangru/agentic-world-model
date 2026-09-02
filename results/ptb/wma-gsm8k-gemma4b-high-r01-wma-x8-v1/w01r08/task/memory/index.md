# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes | supported | iterate | accuracy=0.0333 |  |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/ckpts/exp02/final |
| exp-03 | 1.3 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.7933 | /home/ben/task/ckpts/exp02/final |
| exp-04 | 3.2 | sft | exp-02 | closed | yes (re-locked 1x) | inconclusive | iterate |  |  |
| exp-05 | 4.4 | sft | exp-02 | closed | yes | inconclusive | iterate | accuracy=0.8 | /home/ben/task/ckpts/exp05/final |
| exp-06 | 7.2 | other | exp-05 | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.8 | /home/ben/task/ckpts/exp05/final |
