# Experiment index

One line per card; newest last. Read this before opening a new card.

| card | h | family | parent | status | locked | verdict | decision | best | checkpoint |
|---|---:|---|---|---|---|---|---|---|---|
| exp-01 | 0.15 | other | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.04 | /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| exp-02 | 0.6 | sft | base_model | closed | yes (re-locked 1x) | supported | adopt | accuracy=0.6933 | /home/ben/task/ckpts/exp-02/final |
| exp-03 | 3.3 | decode-config | exp-02 | closed | yes | inconclusive | adopt | accuracy=0.7133 | /home/ben/task/ckpts/exp-03_greedy |
| exp-04 | 3.9 | rft | exp-02 | closed | yes | supported | adopt | accuracy=0.76 | /home/ben/task/ckpts/exp-04/checkpoint-1484 |
| exp-05 | 5.8 | merge | exp-04 | closed | yes | inconclusive | iterate | accuracy=0.7533 | /home/ben/task/ckpts/exp-05_soup |
| exp-06 | 6.05 | other | exp-04 | closed | yes | supported | adopt | accuracy=0.7407 | /home/ben/task/ckpts/exp-05_soup |
| exp-07 | 6.6 | sft | exp-02 | closed | yes | contradicted | reject | accuracy=0.7354 | /home/ben/task/ckpts/exp-07/final |
| exp-08 | 8.05 | merge | exp-04 | closed | yes | contradicted | reject | accuracy=0.7533 | /home/ben/task/ckpts/exp-08_soup4 |
