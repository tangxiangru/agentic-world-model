# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | decode-config | base_model | closed | yes | supported | iterate | accuracy=0.0667 |  |
| exp-02 | 0.42 | sft | base_model | closed | yes | supported | adopt | accuracy=0.5333 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 2.08 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.6733 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 3.43 | rft | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7 | /home/ben/task/ckpts/exp-04-greedy |
| exp-05 | 4.82 | sft | base_model | closed | yes | contradicted | adopt | accuracy=0.7267 | /home/ben/task/ckpts/exp-05-greedy |
| exp-06 | 7.02 | decode-config | exp-04 | closed | yes | contradicted | adopt | accuracy=0.706 | /home/ben/task/ckpts/exp-05-greedy |
| exp-07 | 7.1 | merge | exp-05 | closed | yes | inconclusive | iterate | accuracy=0.706 | /home/ben/task/ckpts/exp-07-soup |
| exp-08 | 7.35 | other | exp-05 | closed | yes | supported | adopt | accuracy=0.7467 | /home/ben/task/final_model |
