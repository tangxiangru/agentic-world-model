# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.3 | other | base_model | closed | yes (re-locked 1x) | supported | iterate | accuracy=0.07333333333333333 |  |
| exp-02 | 1.0 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5933333333333334 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.9 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6533333333333333 | /home/ben/task/ckpts/exp-02-greedy |
| exp-04 | 3.3 | sft | exp-02 | closed | yes | supported | iterate | accuracy=0.6933333333333334 | /home/ben/task/ckpts/exp-04/final |
| exp-05 | 4.0 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.6933333333333334 | /home/ben/task/ckpts/exp-02-ck1800-greedy |
| exp-06 | 4.7 | merge | exp-02 | closed | yes | contradicted | reject | accuracy=0.66 | /home/ben/task/ckpts/exp-06-soup |
| exp-07 | 5.0 | other | exp-02 | closed | yes | contradicted | adopt | accuracy=0.6914329037149356 | /home/ben/task/ckpts/exp-04-final-greedy |
| exp-08 | 6.8 | sft | exp-04 | closed | yes | supported | adopt | accuracy=0.7266666666666667 | /home/ben/task/final_model |
