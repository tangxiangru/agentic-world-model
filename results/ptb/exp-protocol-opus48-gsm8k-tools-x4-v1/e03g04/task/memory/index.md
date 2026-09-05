# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.4 | other | base_model | closed | yes | supported | iterate | accuracy=0.0467 |  |
| exp-02 | 0.75 | sft | base_model | closed | yes | supported | adopt | accuracy=0.4 | /home/ben/task/ckpts/exp-02 |
| exp-03 | 1.2 | decode-config | exp-02 | closed | yes | supported | adopt | accuracy=0.527 | /home/ben/task/ckpts/exp-03-greedy |
| exp-04 | 1.3 | sft | base_model | closed | yes | contradicted | reject | accuracy=0.04 | /home/ben/task/ckpts/exp-04 |
| exp-05 | 3.1 | other | exp-02 | closed | yes | supported | adopt | accuracy=0.852 | /home/ben/task/data/rft_correct.jsonl |
| exp-06 | 3.3 | rft | base_model | closed | yes | contradicted | iterate | accuracy=0.307 | /home/ben/task/ckpts/exp-06 |
| exp-07 | 4.7 | rft | base_model | closed | yes | contradicted | iterate | accuracy=0.46 | /home/ben/task/ckpts/exp-07 |
| exp-08 | 5.5 | rft | base_model | closed | yes | supported | adopt | accuracy=0.567 | /home/ben/task/ckpts/exp-08 |
| exp-09 | 7.7 | other | exp-08 | closed | yes | supported | adopt | accuracy=0.5733 | /home/ben/task/final_model |
| exp-10 | 7.8 | other | exp-08 | closed | yes | supported | adopt | accuracy=0.58 | /home/ben/task/final_model |
